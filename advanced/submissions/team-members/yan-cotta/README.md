# FinResearch AI - Advanced Track

## Multi-Agent Financial Research System

A production-grade, hierarchical multi-agent system for automated financial research, built with CrewAI. This implementation serves as the gold standard reference for the FinResearch AI project.

**Key Features:**
- 🌐 **Web Interface** - Modern React UI with real-time feedback
- 🚀 **Parallel Execution** - Research and Analysis run simultaneously for faster results
- 💬 **Interactive Mode** - CLI conversational interface with context persistence
- 🧠 **Memory System** - ChromaDB-powered context sharing between agents
- 📊 **Professional Reports** - Markdown reports with validation
- 🐳 **Docker Ready** - Full-stack monolith deployment

---

## System Architecture

```
                              INPUT
                    ┌─────────────────────────────┐
                    │   Web UI (React + Vite)     │
                    │   FastAPI REST Backend      │
                    │   CLI (main.py)             │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   FastAPI (src/api.py)      │
                    │   POST /api/research        │
                    │   Async Thread Pool         │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   FinResearchCrew           │
                    │     (crew.py)               │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   Manager Agent             │
                    │   (Orchestrator)            │
                    └────────────┬────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │          PARALLEL EXECUTION            │
            │                                        │
            ▼                                        ▼
     ┌─────────────┐                          ┌─────────────┐
     │ Researcher  │     (async)              │  Analyst    │
     │ Qualitative │◄───────────────────────►│ Quantitative│
     └──────┬──────┘                          └──────┬──────┘
            │                                        │
            │         ┌─────────────┐                │
            └────────►│ ChromaDB    │◄───────────────┘
                      │ (Memory)    │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  Reporter   │
                      │  (Waits for │
                      │   both)     │
                      └──────┬──────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  Markdown Report    │
                    │   (./outputs/)      │
                    └─────────────────────┘
```

---

## Directory Structure

```
yan-cotta/
├── main.py                 # CLI entry point
├── run_dev.sh              # Development server script
├── verify_full_run.py      # End-to-end verification script
├── pyproject.toml          # Project metadata and dependencies
├── Dockerfile              # Full-stack container configuration
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git exclusions
├── README.md               # This documentation
│
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI REST backend
│   ├── crew.py             # Crew orchestration logic
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py     # Pydantic settings management
│   │   ├── agents.yaml     # Agent personas and prompts
│   │   └── tasks.yaml      # Task templates
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py         # Base factory and utilities
│   │   ├── manager.py      # Manager agent (orchestrator)
│   │   ├── researcher.py   # Researcher agent (qualitative)
│   │   ├── analyst.py      # Analyst agent (quantitative)
│   │   └── reporter.py     # Reporter agent (synthesis + ReportOutput model)
│   │
│   └── tools/
│       ├── __init__.py
│       ├── base.py         # Base tool classes
│       ├── financial_data.py   # Yahoo Finance wrapper
│       ├── news_search.py      # DuckDuckGo wrapper
│       └── memory.py           # ChromaDB memory tool
│
├── frontend/               # React Web Interface
│   ├── package.json        # Node.js dependencies
│   ├── vite.config.js      # Vite build configuration
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   ├── index.html          # HTML entry point
│   └── src/
│       ├── main.jsx        # React entry point
│       ├── App.jsx         # Main application component
│       └── index.css       # Tailwind CSS styles
│
├── outputs/                # Generated reports (gitignored)
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    └── test_financial_tool.py  # Tool unit tests
```

---

## Quick Start

### Option 1: Web Interface (Recommended)

```bash
# Navigate to project
cd advanced/submissions/team-members/yan-cotta

# Install Python dependencies
pip install -e .

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start development servers
./run_dev.sh
```

Open http://localhost:5173 in your browser.

### Option 2: Docker (Production)

```bash
# Build the full-stack image
docker build -t finresearch-ai:latest .

# Run web server
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... finresearch-ai:latest

# Run CLI mode
docker run -e OPENAI_API_KEY=sk-... finresearch-ai:latest cli AAPL --sequential
```

### Option 3: CLI Only

```bash
# Research a stock
python main.py AAPL --sequential

# Interactive mode
python main.py -i
```

---

## Web Interface

### Features

The React web interface provides:

- **Ticker Input** - Enter any stock symbol (AAPL, TSLA, etc.)
- **Quick Select** - Popular ticker buttons for fast access
- **Settings Panel**
  - Reset Memory toggle (clear previous context)
  - Model Provider selection (OpenAI/Groq)
- **Real-time Progress** - Agent activity visualization
- **Report Viewer** - Rendered markdown with syntax highlighting
- **Download** - Save reports as markdown files

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research` | POST | Execute research (sync) |
| `/api/research/async` | POST | Start research (async) |
| `/api/status/{task_id}` | GET | Poll async task status |
| `/api/health` | GET | Health check |
| `/api/memory` | DELETE | Clear ChromaDB memory |
| `/api/docs` | GET | Swagger documentation |

### Request Example

```json
POST /api/research
{
  "ticker": "AAPL",
  "company_name": "Apple Inc",
  "reset_memory": false,
  "model_provider": "openai",
  "sequential_mode": true
}
```

### Response Example

```json
{
  "success": true,
  "ticker": "AAPL",
  "report": "# Financial Research Report: AAPL\n\n## Executive Summary...",
  "logs": ["Starting research...", "Executing agent workflow...", "Research completed!"],
  "duration_seconds": 42.5,
  "error": null
}
```

---

## Module Overview

### Configuration (`src/config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Pydantic-based settings with environment variable support |
| `agents.yaml` | Agent roles, goals, and backstories (prompts) |
| `tasks.yaml` | Task description templates with placeholders |

### Tools (`src/tools/`)

| Tool | Data Source | Purpose |
|------|-------------|---------|
| `FinancialDataTool` | Yahoo Finance | Stock prices, valuation metrics, fundamentals |
| `NewsSearchTool` | DuckDuckGo | News articles with source verification |
| `MemoryTool` | ChromaDB | Persistent vector memory for agent collaboration |

### Agents (`src/agents/`)

| Agent | Role | Temperature | Tools |
|-------|------|-------------|-------|
| Manager | Orchestration and delegation | 0.1 | None (hierarchical) |
| Researcher | Qualitative analysis | 0.7 | News, Memory |
| Analyst | Quantitative analysis | 0.0 | Financial, Memory |
| Reporter | Report synthesis | 0.5 | Memory |

### API Backend (`src/api.py`)

FastAPI-based REST backend with:
- Thread pool executor for async CrewAI execution
- CORS middleware for frontend development
- Static file serving for production deployment
- Pydantic models for request/response validation

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- OpenAI API key

### Backend Installation

```bash
cd advanced/submissions/team-members/yan-cotta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Frontend Installation

```bash
cd frontend
npm install
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Optional overrides
FINRESEARCH_MANAGER_MODEL=gpt-4o-mini
FINRESEARCH_WORKER_MODEL=gpt-3.5-turbo
FINRESEARCH_LOG_LEVEL=INFO
FINRESEARCH_OUTPUT_DIR=./outputs
```

---

## Development

### Running Development Servers

```bash
# Option 1: Use the dev script
./run_dev.sh

# Option 2: Run separately
# Terminal 1 - Backend
uvicorn src.api:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### Building for Production

```bash
# Build frontend
cd frontend && npm run build && cd ..

# The built files are in frontend/dist/
# FastAPI will serve them automatically
```

### Docker Build

```bash
# Build full-stack image
docker build -t finresearch-ai:latest .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/outputs:/app/outputs \
  finresearch-ai:latest
```

---

## CLI Usage

### Command Line Interface

```bash
# Research Apple Inc
python main.py AAPL

# Research with company name
python main.py TSLA --name "Tesla Inc"

# Use sequential process (simpler, for debugging)
python main.py MSFT --sequential

# Custom output filename
python main.py GOOGL --output google_research.md

# Verbose mode (see agent reasoning)
python main.py NVDA --verbose
```

### Interactive Mode

```bash
# Start interactive mode
python main.py --interactive
# or
python main.py -i
```

Interactive commands:
- `AAPL` - Research a ticker
- `context AAPL` - Show previous context
- `status` - Session summary
- `clear` - Reset session
- `help` - Show commands
- `exit` - Quit

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run verification script
python verify_full_run.py --dry-run

# Test with specific ticker
python verify_full_run.py --ticker AAPL
```

---

## Troubleshooting

### OPENAI_API_KEY not configured

Ensure your `.env` file is in the project root and contains a valid key.

### Frontend not loading

1. Check that `npm install` was run in `frontend/`
2. Verify Vite dev server is running on port 5173
3. Check browser console for CORS errors

### API returning 500 errors

Check the FastAPI logs for detailed error messages:
```bash
uvicorn src.api:app --reload --log-level debug
```

### Agent seems stuck

Use sequential mode for simpler execution:
```bash
python main.py AAPL --sequential
```

---

## License

MIT License - See project root for details.

---

**Author:** Yan Cotta

**Version:** 1.0.0

**Last Updated:** December 2025
