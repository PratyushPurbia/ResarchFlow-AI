# 🔬 ResearchFlow AI

A multi-agent research assistant powered by **LangGraph** that automatically plans, researches, and writes comprehensive reports on any topic.

## Architecture

```
User Query → Planner → [Researcher → Tools → Synthesize] × N → Writer → Report
```

**Three specialized agents:**
- **Planner** — Breaks down research queries into focused tasks
- **Researcher** — Uses web search tools to gather information, synthesizes findings
- **Writer** — Compiles all findings into a structured, professional report

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) — Multi-agent orchestration with stateful graphs
- [LangChain](https://github.com/langchain-ai/langchain) — LLM framework
- [Groq](https://groq.com/) — Fast LLM inference (free tier)
- [Tavily](https://tavily.com/) — Web search API
- [Streamlit](https://streamlit.io/) — Frontend UI

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/PratyushPurbia/ResarchFlow-AI.git
cd ResarchFlow-AI
```

### 2. Set up Python environment

```bash
# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
- **GROQ_API_KEY** — Get free from [console.groq.com/keys](https://console.groq.com/keys)
- **TAVILY_API_KEY** — Get free from [tavily.com](https://tavily.com/)

### 4. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Features

- 🧠 **Intelligent Planning** — Automatically decomposes complex queries into research tasks
- 🔍 **Web Research** — Real-time web search via Tavily API
- 📊 **Live Progress** — See plan, search queries, findings, and report generation in real-time
- 📄 **Professional Reports** — Structured markdown reports with sections
- 📥 **Download** — Export reports as markdown files
- ⚡ **Fast** — All inference on Groq's LPU for sub-second responses
- 🆓 **Free** — Uses only free-tier APIs

## Token Optimization

The pipeline is optimized for free-tier usage:
- Tool call loops capped at 1 round per task
- Thinking tokens stripped from all intermediate messages
- Research messages reset between tasks (no context accumulation)
- Retry logic for transient network errors

## Project Structure

```
├── app.py                    # Streamlit frontend + full pipeline
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys
├── .gitignore
└── ResearchFlow_Phase1/      # Development notebooks
    ├── phase1.ipynb          # Initial prototype
    └── phase2.ipynb          # Iterative development
```

## License

MIT
