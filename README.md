# Verified Research Engine

Verified Research Engine (VRE) writes sourced research briefs. You give it a topic; it splits the topic into questions, reads web pages for each one, and writes a brief in which every factual sentence cites a passage copied word for word from the page it came from.

> **Status: in development. Phase 3 of 12 is complete.**
> Today a topic goes in and a cited brief comes out, from public web pages read through the project's own MCP server, with every sub-question researched at the same time. Every cited passage is checked in code to occur on its page; whether each sentence is supported by its passage is checked from Phase 5. Your own PDF documents, human approval, and the API arrive in later phases (see [Roadmap](#roadmap)).

When finished, VRE will differ from a general AI report writer in three ways:

- **It reads your files as well as the web**, and reports where the two disagree.
- **It checks its own citations.** A claim whose source does not support it is rewritten, removed, or labelled unverified.
- **Nothing is final until a person approves it**, and every run records what it did and what it used.

## Sample output

The same topic, at the end of each phase, so the two can be compared:

| Brief | What it shows |
| --- | --- |
| [Phase 3: due diligence on Symbotic](notebooks/phase_03_brief.md) | The six sub-questions researched at the same time; 14 findings, 0 sub-questions without evidence. Four runs of this topic took 43.7 to 84.6 s, against 99.4 s in Phase 2 |
| [Phase 2: the same topic](notebooks/phase_02_brief.md) | Passages chosen from pages the system read, each checked to be on its page; sub-questions researched one after another |
| [Phase 1: the same topic](notebooks/phase_01_brief.md) | Passages were the snippets a search engine returned, some only loosely related to the question |

Each brief's header lists the prompt versions, the models, and the run numbers that produced it. [`notebooks/phase_03.ipynb`](notebooks/phase_03.ipynb) runs it, shows where each part of the run spent its time, compares the three phases, and ends with the **phase record**: the problems met while building Phase 3 and what was decided about each.

## How it works (Phase 3)

```text
topic
  |
  v
planner          one model call: splits the topic into up to 6 sub-questions (Q1..Q6)
  |
  |  one branch per sub-question, all running at the same time
  v
web_researcher   x 6, each through the MCP toolbox only:
  |                web_search  -> 5 results
  |                fetch_page  -> all 5 read at once, first 3 readable pages kept
  |                P-02        -> the model proposes passages that answer the question
  |                checks      -> a passage is kept only if it is on its page, word for word
  v
research_check   stops the run if every search failed, instead of writing from nothing
  |
  v
writer           one outline call, then one call per section, each given only that
  |              section's findings; every citation is checked against them in code
  v
brief            Markdown, with one line per source page it actually cites
```

```text
graph (researcher)  --MCP over HTTP-->  search-mcp  --> Tavily (search)
   McpToolbox                           web_search      the page itself (fetch_page)
```

The graph never imports a search or page-reading library: it reaches the web only through `search-mcp`, a FastMCP server in this repository, and a test fails if any file in `graph/` imports one. One run is one trace in LangSmith, with every model call and every tool call, its prompt version, and its inputs and outputs.

### Principles the code follows

| Principle | Where |
| --- | --- |
| Every model decision returns a validated Pydantic object, never free text | `graph/schemas.py` |
| Every model call goes through one gateway: retries, trace tags, and the limits below | `graph/llm.py` |
| Each model has its own rate limit, at 80% of the provider's, shared by the whole process; a refused request is retried after a randomised wait | `graph/llm.py` |
| A run may send at most 30 model requests, retries included, so a loop cannot keep spending | `graph/llm.py`, `graph/supervisor.py` |
| Sub-questions are researched in parallel branches; results are kept in the planner's order, whatever order the branches finish in | `graph/supervisor.py`, `graph/state.py` |
| Ids (`Q2`, `Q2-F1`) are assigned by code, never by the model | `graph/nodes/planner.py`, `graph/extraction.py` |
| The model proposes; code verifies: a passage is kept only if it is on its page | `graph/extraction.py` |
| Tools are reached only through MCP; the graph imports no tool library | `graph/mcp_client.py`, `tests/unit/test_graph_imports.py` |
| A page that cannot be read is skipped; a search server that is gone stops the run | `graph/nodes/researcher.py` |
| The search server refuses private and internal addresses, on every redirect | `mcp_servers/search_mcp/url_safety.py` |
| The whole graph is tested end to end with a scripted model and a fake toolbox, with no network; replies to parallel calls are keyed to their question, so the tests do not depend on timing | `tests/` |

## Stack

| Part | Choice |
| --- | --- |
| Models | Mistral: `ministral-8b-latest` for decisions and extraction, `ministral-14b-latest` for writing |
| Orchestration | LangGraph |
| Tools | `search-mcp`, a FastMCP server (MCP over HTTP), reached through FastMCP's client |
| Web search, page text | Tavily, inside `search-mcp`; pages read and cleaned with `trafilatura` |
| Tracing | LangSmith |
| Environment | Python 3.12, `uv`, `ruff`, `pytest` |
| Coming | FastAPI with server-sent events, two more MCP servers, SQLite checkpoints, Docker |

Everything runs on free tiers, and every account has a spending cap: no payment card on any model or tracing account, and pay as you go switched off for search, so a free tier stops at its limit instead of charging.

## Run it

You need Python 3.12 (installed by `uv`), [`uv`](https://docs.astral.sh/uv/), and free API keys for Mistral, Tavily, and LangSmith. Commands are for Windows PowerShell.

```powershell
git clone https://github.com/HuMzAHCodes/verified-research-engine.git
cd verified-research-engine
uv sync
uv run nbstripout --install
Copy-Item .env.example .env
notepad .env
```

Fill in the three keys in `.env`. The file is ignored by git; never commit it.

**Tests that need no keys and no network** (the same ones CI runs on every push):

```powershell
uv run pytest -m "not live"
```

**Start the search server** in its own terminal, and leave it running while you work:

```powershell
uv run python -m mcp_servers.search_mcp.server
```

It listens on `http://127.0.0.1:8765/mcp`, on this machine only. Stop it with `Ctrl+C`.

**One real brief:** with the server running, open `notebooks/phase_03.ipynb` in VS Code, choose the project's `.venv` as the kernel, and run all cells. A run takes about 45 to 90 seconds, depending on how busy the model provider is, makes up to 14 model requests (at most 30 are allowed) and 6 searches, prints the LangSmith link to its trace, and writes `notebooks/phase_03_brief.md`.

**Live checks** against the real APIs and the running server (they spend free-tier quota):

```powershell
uv run pytest -m live -s
```

## Keeping secrets out of this repository

- API keys are read only from `.env`, and are never printed. The graph reads its settings in one place (`settings.py`); `search-mcp` reads only the search key, in its own `config.py`.
- `nbstripout` removes notebook outputs at commit time, so nothing a cell printed is committed.
- A test (`tests/unit/test_no_secrets.py`) runs in CI and fails if any tracked file contains a key-shaped string, an actual key value, a tracked `.env`, or a notebook with outputs.

## Project layout

```text
settings.py              the graph's only reader of environment variables
graph/
  llm.py                 model gateway
  mcp_client.py          McpToolbox: the graph's only way to reach tools
  extraction.py          P-02 and the word-for-word checks
  schemas.py             what each model call must return
  state.py               what a run keeps between steps
  supervisor.py          builds the graph
  report.py              renders a brief as Markdown
  prompts/               prompt text, one module per prompt group, each with a version
  nodes/                 planner, researcher, writer
mcp_servers/
  search_mcp/            web_search and fetch_page; its own settings; the address check
tests/
  unit/                  scripted tests: no keys, no network
  live/                  real API checks, run on request
notebooks/
  phase_03.ipynb         one real run, timed by stage and branch, and the Phase 3 record
  phase_03_brief.md      the brief it wrote
  phase_02.ipynb         the Phase 2 record (re-running it now uses the Phase 3 graph)
  phase_02_brief.md      the Phase 2 brief on the same topic
  phase_01.ipynb         the Phase 1 record (no longer runs: its search client was removed)
  phase_01_brief.md      the Phase 1 brief on the same topic
```

## Roadmap

| Phase | Delivers | Status |
| --- | --- | --- |
| 1 | Planner, researcher, and writer produce a cited brief from a topic | Done |
| 2 | Web research through a custom MCP server; only passages copied word for word are kept | Done |
| 3 | One researcher per sub-question, in parallel, with a rate limit per model and a request budget per run | Done |
| 4 | Your own PDFs as a second source, through a Self-RAG subgraph | Next |
| 5 | Every claim checked against its cited passage; failed claims revised | Planned |
| 6 | Human approval before release; runs survive a restart | Planned |
| 7 | Preferences that carry over between runs | Planned |
| 8 | REST API (FastAPI) | Planned |
| 9 | Background jobs and live progress | Planned |
| 10 | Metrics from traces, evaluation, a user trial | Planned |
| 11 | Docker, cloud deployment, a remote MCP server | Planned |
| 12 | Optional extras | Optional |
