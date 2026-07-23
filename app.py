"""
ResearchFlow AI - Multi-Agent Research Assistant
Streamlit Frontend with Full Progress UI
"""

import streamlit as st
import ast
import re
import time
import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch

load_dotenv(override=True)

# --- Configuration ---
MODELS = {
    "planner": "qwen/qwen3.6-27b",
    "researcher": "qwen/qwen3.6-27b",
    "writer": "qwen/qwen3.6-27b",
}

DEV_CONFIG = {
    "max_tasks": 5,
    "max_finding_words": 150,
    "max_search_results": 3,
    "max_tool_rounds": 1,
    "max_report_words": 1500,
}

# --- LLM Setup ---
@st.cache_resource
def get_llms():
    import os
    load_dotenv(override=True)

    planner_llm = init_chat_model(
        model=MODELS["planner"],
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    researcher_llm = init_chat_model(
        model=MODELS["researcher"],
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    writer_llm = init_chat_model(
        model=MODELS["writer"],
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,
    )
    return planner_llm, researcher_llm, writer_llm


# --- State ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    research_messages: list
    plan: list[str]
    findings: list[dict]
    current_task: int
    tool_rounds: int
    report: str


# --- Prompts ---
PLANNER_PROMPT = """
You are a research planner.

Your job is to break a user query into research tasks.

Rules:

- Create the minimum number of tasks required.
- Avoid overlapping tasks.
- Each task should investigate a different aspect.
- Every task must be self-contained.
- Every task must explicitly include the subject being researched.
- Never create generic tasks.

For simple factual questions:
Create 1 task.

For comparison questions:
Create 3 tasks.

For deep research:
Create up to 5 tasks.

Return ONLY a Python list of strings. No explanation. No markdown.

Example output:
["Compare LangGraph and CrewAI architecture", "Compare LangGraph and CrewAI strengths and weaknesses", "Compare LangGraph and CrewAI use cases"]
"""

RESEARCHER_PROMPT = """
You are a research specialist.

Research ONE task.

Rules:

- Use tools whenever external information is needed.
- Read tool results carefully.
- Extract only the most relevant information.
- Synthesize information from multiple sources.
- Focus only on the assigned task.
- Do not write a report.
- Do not write introductions or conclusions.
- Do not create sections.
- Do not copy source text verbatim.
- Use tools at most ONCE. Do not make follow-up searches.
- If evidence is insufficient, explicitly state what is missing.
- Do NOT include any thinking, reasoning, or chain-of-thought in your output.

Output format:
- Finding 1
- Finding 2
- Finding 3
- Finding 4

Maximum {max_finding_words} words.
""".format(max_finding_words=DEV_CONFIG["max_finding_words"])

WRITER_PROMPT = """
You are an expert report writer.

You will receive multiple research findings.

Create a coherent, detailed report using ONLY those findings.

Rules:

- Do not invent information.
- Do not include chain-of-thought or thinking.
- Do not mention missing information unless present in findings.
- Synthesize findings into a readable, professional report.
- Avoid repetition.
- Be thorough and detailed.

Format:

# Title

## Overview

## Key Findings

## Detailed Analysis

## Conclusion
"""


# --- Graph Nodes ---
def build_graph(planner_llm, researcher_llm, writer_llm, progress_callback=None):
    """Build the research graph with all nodes and edges."""
    import os

    tavily_tool = TavilySearch(max_results=DEV_CONFIG["max_search_results"], api_key=os.getenv("TAVILY_API_KEY"))

    # Wrap tavily to only accept 'query' param (prevents models from hallucinating extra params)
    from langchain_core.tools import tool as tool_decorator

    @tool_decorator
    def web_search(query: str) -> str:
        """Search the web for current information on a topic. Use this to find facts, comparisons, and recent data."""
        return tavily_tool.invoke({"query": query})

    tools = [web_search]
    researcher_with_tools = researcher_llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def planner(state: State):
        if progress_callback:
            progress_callback("planner", "Planning research tasks...")

        user_query = state["messages"][-1].content
        response = planner_llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_query),
        ])
        content = response.content

        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Extract list even if model adds extra text
        list_match = re.search(r'\[.*\]', content, re.DOTALL)
        if list_match:
            content = list_match.group(0)

        plan = ast.literal_eval(content)

        # Cap tasks
        plan = plan[:DEV_CONFIG["max_tasks"]]

        if progress_callback:
            progress_callback("planner_done", plan)

        return {"plan": plan, "current_task": 0}

    def prepare_task(state: State):
        task = state["plan"][state["current_task"]]
        if progress_callback:
            progress_callback("prepare_task", {
                "task_index": state["current_task"],
                "task": task,
                "total": len(state["plan"]),
            })
        return {
            "research_messages": [
                SystemMessage(content=RESEARCHER_PROMPT),
                HumanMessage(content=task),
            ],
            "tool_rounds": 0,
        }

    def researcher(state: State):
        if progress_callback:
            progress_callback("researcher", "Researching...")

        response = researcher_with_tools.invoke(state["research_messages"])

        if response.tool_calls:
            if progress_callback:
                queries = [tc["args"].get("query", "") for tc in response.tool_calls]
                progress_callback("tool_calls", queries)

            # Strip thinking, keep only tool calls
            clean = AIMessage(content="", tool_calls=response.tool_calls)
            return {
                "research_messages": state["research_messages"] + [clean],
                "tool_rounds": state.get("tool_rounds", 0) + 1,
            }

        # Normal text response
        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        clean_response = AIMessage(content=content)
        return {
            "research_messages": state["research_messages"] + [clean_response],
        }

    def research_router(state: State):
        last_message = state["research_messages"][-1]
        max_tool_rounds = DEV_CONFIG.get("max_tool_rounds", 1)

        if state.get("tool_rounds", 0) >= max_tool_rounds:
            return "synthesize"

        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tools"

        return "save_finding"

    def synthesize(state: State):
        if progress_callback:
            progress_callback("synthesize", "Synthesizing findings...")

        task = state["plan"][state["current_task"]]
        synthesis_prompt = f"""Based on the search results above, synthesize your findings for this task: {task}

Output ONLY bullet points. Maximum {DEV_CONFIG['max_finding_words']} words. Do not use tools."""

        messages = state["research_messages"] + [HumanMessage(content=synthesis_prompt)]

        content = ""
        for attempt in range(3):
            try:
                response = researcher_llm.invoke(messages)
                content = response.content
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if not content:
            content = "Insufficient data to synthesize findings for this task."

        clean_response = AIMessage(content=content)
        return {
            "research_messages": state["research_messages"] + [clean_response],
        }

    def save_finding(state: State):
        task = state["plan"][state["current_task"]]
        final_answer = state["research_messages"][-1].content

        # Strip thinking from final answer if it leaked through
        if "</think>" in final_answer:
            final_answer = final_answer.split("</think>")[-1].strip()
        if "<think>" in final_answer:
            final_answer = final_answer.split("<think>")[0].strip()

        # Fallback if stripping made it empty
        if not final_answer:
            final_answer = "Insufficient data available for this specific task."

        if progress_callback:
            progress_callback("finding_saved", {
                "task": task,
                "content": final_answer,
                "task_index": state["current_task"],
            })

        return {
            "findings": state.get("findings", []) + [{"task": task, "content": final_answer}],
            "current_task": state["current_task"] + 1,
            "research_messages": [],
        }

    def router(state: State):
        if state["current_task"] < len(state["plan"]):
            return "prepare_task"
        return "writer"

    def writer(state: State):
        if progress_callback:
            progress_callback("writer", "Writing final report...")

        findings_text = "\n\n".join(
            f"TASK: {item['task']}\n\n{item['content']}"
            for item in state["findings"]
        )

        response = writer_llm.invoke([
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=f"Research Findings:\n\n{findings_text}"),
        ])

        report = response.content
        if "</think>" in report:
            stripped = report.split("</think>")[-1].strip()
            report = stripped if stripped else report.split("</think>")[0].replace("<think>", "").strip()

        if progress_callback:
            progress_callback("done", report)

        return {"report": report}

    def tools_wrapper(state: State):
        if progress_callback:
            progress_callback("tools", "Executing search...")
        try:
            result = tool_node.invoke({"messages": state["research_messages"]})
            return {
                "research_messages": state["research_messages"] + [result["messages"][-1]],
            }
        except Exception as e:
            # If tool execution fails, add error as tool message so flow continues
            from langchain_core.messages import ToolMessage
            last_msg = state["research_messages"][-1]
            error_msgs = []
            if hasattr(last_msg, "tool_calls"):
                for tc in last_msg.tool_calls:
                    error_msgs.append(ToolMessage(
                        content=f"Search failed: {str(e)[:100]}",
                        tool_call_id=tc["id"],
                    ))
            return {
                "research_messages": state["research_messages"] + error_msgs,
            }

    # --- Build Graph ---
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("prepare_task", prepare_task)
    builder.add_node("save_finding", save_finding)
    builder.add_node("researcher", researcher)
    builder.add_node("synthesize", synthesize)
    builder.add_node("writer", writer)
    builder.add_node("tools", tools_wrapper)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "prepare_task")
    builder.add_edge("prepare_task", "researcher")
    builder.add_conditional_edges(
        "researcher",
        research_router,
        {"tools": "tools", "save_finding": "save_finding", "synthesize": "synthesize"},
    )
    builder.add_edge("tools", "researcher")
    builder.add_edge("synthesize", "save_finding")
    builder.add_conditional_edges(
        "save_finding",
        router,
        {"prepare_task": "prepare_task", "writer": "writer"},
    )
    builder.add_edge("writer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# --- Streamlit UI ---
st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 ResearchFlow AI")
st.markdown("*Multi-agent research assistant powered by LangGraph*")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Planner:** {MODELS['planner']} (Groq)")
    st.markdown(f"**Researcher:** {MODELS['researcher']} (Groq)")
    st.markdown(f"**Writer:** {MODELS['writer']} (Groq)")
    st.divider()
    st.markdown(f"Max tasks: {DEV_CONFIG['max_tasks']}")
    st.markdown(f"Max tool rounds: {DEV_CONFIG['max_tool_rounds']}")
    st.markdown(f"Max finding words: {DEV_CONFIG['max_finding_words']}")

# Initialize session state
if "research_running" not in st.session_state:
    st.session_state.research_running = False
if "report" not in st.session_state:
    st.session_state.report = None
if "progress_log" not in st.session_state:
    st.session_state.progress_log = []
if "findings" not in st.session_state:
    st.session_state.findings = []
if "plan" not in st.session_state:
    st.session_state.plan = []

# Input
query = st.text_input(
    "🔍 What would you like to research?",
    placeholder="e.g., Compare LangGraph and CrewAI",
    disabled=st.session_state.research_running,
)

col1, col2 = st.columns([1, 5])
with col1:
    start_btn = st.button(
        "🚀 Research",
        disabled=st.session_state.research_running or not query,
        type="primary",
    )
with col2:
    if st.button("🗑️ Clear", disabled=st.session_state.research_running):
        st.session_state.report = None
        st.session_state.progress_log = []
        st.session_state.findings = []
        st.session_state.plan = []
        st.rerun()

if start_btn and query:
    st.session_state.research_running = True
    st.session_state.report = None
    st.session_state.progress_log = []
    st.session_state.findings = []
    st.session_state.plan = []

    progress_container = st.container()
    status_placeholder = st.empty()

    with progress_container:
        progress_area = st.expander("📊 Research Progress", expanded=True)

    def progress_callback(event, data):
        """Handle progress updates from the graph."""
        if event == "planner":
            status_placeholder.info("🧠 " + data)
        elif event == "planner_done":
            st.session_state.plan = data
            with progress_area:
                st.markdown("### 📋 Research Plan")
                for i, task in enumerate(data, 1):
                    st.markdown(f"{i}. {task}")
        elif event == "prepare_task":
            status_placeholder.info(
                f"📝 Task {data['task_index']+1}/{data['total']}: {data['task']}"
            )
        elif event == "researcher":
            pass
        elif event == "tool_calls":
            with progress_area:
                st.markdown(f"🔎 Searching: *{', '.join(data[:2])}*...")
        elif event == "tools":
            pass
        elif event == "synthesize":
            status_placeholder.info("🔄 " + data)
        elif event == "finding_saved":
            st.session_state.findings.append(data)
            with progress_area:
                st.markdown(f"✅ **Finding {data['task_index']+1}:** {data['task']}")
                st.markdown(f"> {data['content'][:200]}...")
        elif event == "writer":
            status_placeholder.info("✍️ " + data)
        elif event == "done":
            st.session_state.report = data
            status_placeholder.success("✅ Research complete!")

    try:
        planner_llm, researcher_llm, writer_llm = get_llms()
        graph = build_graph(planner_llm, researcher_llm, writer_llm, progress_callback)

        thread_id = str(uuid.uuid4())
        result = graph.invoke(
            {
                "messages": [("user", query)],
                "findings": [],
                "current_task": 0,
                "tool_rounds": 0,
                "research_messages": [],
                "plan": [],
                "report": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        if result.get("report"):
            st.session_state.report = result["report"]
            status_placeholder.success("✅ Research complete!")
        else:
            status_placeholder.error("⚠️ Report generation failed. Please try again.")

    except Exception as e:
        status_placeholder.error(f"❌ Error: {str(e)}")
    finally:
        st.session_state.research_running = False

# Display report
if st.session_state.report:
    st.divider()
    st.markdown("## 📄 Research Report")
    st.markdown(st.session_state.report)

    # Download button
    st.download_button(
        label="📥 Download Report (Markdown)",
        data=st.session_state.report,
        file_name="research_report.md",
        mime="text/markdown",
    )
