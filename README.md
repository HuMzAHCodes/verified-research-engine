# Verified Research Engine

Verified Research Engine (VRE) writes sourced research briefs. You give it a topic, and optionally your own PDF documents; it splits the topic into questions, answers each one from public web pages, your documents, or both, and writes a brief in which every factual sentence cites a passage copied word for word from the page it came from.

> **Status: in development. Phase 4 of 12 is complete.**
> Today a topic and your PDFs go in and a cited brief comes out. Web pages are read through the project's own MCP server; documents are read by a second MCP server, stored, and searched by a Self-RAG subgraph that checks its own answers; document sources are cited by page. Every cited passage is checked in code to occur on its page; whether each sentence is supported by its passage is checked from Phase 5. Human approval and the API arrive in later phases (see [Roadmap](#roadmap)).

When finished, VRE will differ from a general AI report writer in three ways:

- **It reads your files as well as the web**, and reports where the two disagree.
- **It checks its own citations.** A claim whose source does not support it is rewritten, removed, or labelled unverified.
- **Nothing is final until a person approves it**, and every run records what it did and what it used.

## Sample output

The same topic, at the end of each phase, so the two can be compared:

| Brief | What it shows |
| --- | --- |
| [Phase 4: due diligence on Symbotic, with its annual report](notebooks/phase_04_brief.md) | Three sub-questions answered from the annual report, two from the web, one from both: 9 web and 7 document findings, the document ones cited by page; 35 model requests of a budget of 118 sized to the plan; 42.5 s |
| [Phase 3: due diligence on Symbotic](notebooks/phase_03_brief.md) | The six sub-questions researched at the same time; 14 findings, 0 sub-questions without evidence. Four runs of this topic took 43.7 to 84.6 s, against 99.4 s in Phase 2 |
| [Phase 2: the same topic](notebooks/phase_02_brief.md) | Passages chosen from pages the system read, each checked to be on its page; sub-questions researched one after another |
| [Phase 1: the same topic](notebooks/phase_01_brief.md) | Passages were the snippets a search engine returned, some only loosely related to the question |

Each brief's header lists the prompt versions, the models, and the run numbers that produced it. [`notebooks/phase_04.ipynb`](notebooks/phase_04.ipynb) times a document's ingestion request by request, runs the brief with the annual report, checks every document citation against its page, compares the four phases, and ends with the **phase record**: the problems met while building Phase 4 and what was decided about each.

## How it works (Phase 4)

**Once per document (ingestion):**

```text
your PDF in data/uploads/
  -> pdf-parser-mcp     reads the text of each page (MCP over HTTP)
  -> chunks             at most 600 characters, inside one page, each an exact slice of the page text
  -> mistral-embed      one request per 32 chunks, up to three waiting at once
  -> store              one FAISS index per document, with each chunk's page, in data/vectors/
```

A file already stored is recognised by its fingerprint and not embedded again.

**Every run:**

```text
topic (+ your stored documents)
  |
  v
planner              one model call: up to 6 sub-questions, each routed to web, document, or both;
  |                  the run's request budget is then sized to this plan
  |
  |  one branch per sub-question and source, all running at the same time
  v
web_researcher       web_search -> 5 results, the same page under two addresses read once
  |                  fetch_page -> first 3 readable pages; P-02 proposes passages;
  |                  a passage is kept only if it is on its page, word for word
  |
document_researcher  the Self-RAG subgraph, over the run's documents only:
  |                    write a passage shaped like the answer (HyDE), search with it and the
  |                    question (keyword and meaning, merged), grade the chunks, answer,
  |                    check support and usefulness, revise into quotes if needed, or say
  |                    "not found"; cite only the chunks that hold the supporting quotes
  v
research_check       stops the run if every web search failed, instead of writing from nothing
  |
  v
writer               one outline call, then one call per section, given only that section's
  |                  findings; every citation is checked against them in code
  v
brief                Markdown; web sources cited by address, document sources by page
```

```text
graph (researchers) --MCP over HTTP--> search-mcp      --> Tavily (search), the page itself
ingestion           --MCP over HTTP--> pdf-parser-mcp  --> pypdf, inside data/uploads/ only
```

The graph never imports a search, page-reading, or PDF library: it reaches tools only through MCP, and a test fails if any file in `graph/` imports one. Document text never reaches a web search: the document branch has no search tool, and a test plants a marker in a document and checks every outgoing search and web prompt. One run is one trace in LangSmith.

### Principles the code follows

| Principle | Where |
| --- | --- |
| Every model decision returns a validated Pydantic object, never free text | `graph/schemas.py` |
| Every model call goes through one gateway: retries, trace tags, and the limits below | `graph/llm.py` |
| Each model has its own rate limit, at 80% of the provider's, shared by the whole process; a refused request is retried after a randomised wait | `graph/llm.py` |
| A run's request budget is sized to its plan (every designed cap reached, one schema retry per call), never above 200, so a loop cannot keep spending | `graph/run_budget.py`, `graph/llm.py` |
| A model's limit is read from its replies, not from a console page; a refused request waits for the next minute and logs the limits the provider reports | `graph/llm.py` |
| Sub-questions are researched in parallel branches; results are kept in the planner's order, whatever order the branches finish in | `graph/supervisor.py`, `graph/state.py` |
| Ids (`Q2`, `Q2-F1`) are assigned by code, never by the model | `graph/nodes/planner.py`, `graph/extraction.py` |
| The model proposes; code verifies: a passage is kept only if it is on its page word for word, starting and ending on whole words and whole figures, so a cut-off "$1.2" never stands in for "$1.25 billion" | `graph/extraction.py` |
| A document answer is checked for support and usefulness before it is used, and cites only the chunks that hold its supporting quotes, each an exact slice of its page | `graph/self_rag/`, `graph/nodes/document_researcher.py` |
| A document search sees only the run's own documents | `graph/retrieval.py` |
| Tools are reached only through MCP; the graph imports no tool library | `graph/mcp_client.py`, `tests/unit/test_graph_imports.py` |
| A page that cannot be read is skipped; a search server that is gone stops the run | `graph/nodes/researcher.py` |
| The search server refuses private and internal addresses, on every redirect, including ones hidden inside IPv6 forms or written as numbers | `mcp_servers/search_mcp/url_safety.py` |
| The whole graph is tested end to end with a scripted model, scripted embeddings, a fake toolbox, and a store in a temporary folder, with no network; replies to parallel calls are keyed to their prompt, so the tests do not depend on timing | `tests/` |
| Every phase closes with an outside test review: a reviewer that has seen none of the work reads the code and tests and looks for what they miss. At the Phase 4 close, 393 passing tests and 96% coverage had missed 15 bugs; each was fixed only after a test failed on the old code (501 scripted tests and 98% coverage after) | `scripts/make_review_pack.py` |

## Stack

| Part | Choice |
| --- | --- |
| Models | Mistral: `voxtral-small-2507` for every call (the one model the free plan still allows at 60 requests a minute), `mistral-embed` for document chunks |
| Orchestration | LangGraph, with the Self-RAG subgraph inside the research step |
| Tools | `search-mcp` and `pdf-parser-mcp`, FastMCP servers (MCP over HTTP), reached through FastMCP's client |
| Web search, page text | Tavily, inside `search-mcp`; pages read and cleaned with `trafilatura` |
| Documents | `pypdf` inside `pdf-parser-mcp`; FAISS (one exact index per document); `rank-bm25` (BM25Plus) for keyword search |
| Tracing | LangSmith |
| Environment | Python 3.12, `uv`, `ruff`, `pytest` |
| Coming | FastAPI with server-sent events, one more MCP server, SQLite checkpoints, Docker |

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

**Start both MCP servers**, each in its own window; the script returns once both are listening:

```powershell
.\scripts\start_mcp_servers.ps1
```

`search-mcp` listens on `http://127.0.0.1:8765/mcp` and `pdf-parser-mcp` on `http://127.0.0.1:8766/mcp`, on this machine only. Close a window, or press `Ctrl+C` in it, to stop a server.

**Add a document:** put a PDF with a text layer (not a scan) in `data/uploads/`, which git ignores, and ingest it once, with the servers running:

```powershell
uv run python -c "import asyncio; from app.ingestion import IngestionService; from graph.document_store import DocumentStore; from graph.llm import ModelGateway; result = asyncio.run(IngestionService(ModelGateway(), DocumentStore()).ingest('data/uploads/your-file.pdf')); print(result.document.document_id)"
```

It prints the document's id. The notebook uses Symbotic's annual report (Form 10-K), saved as `data/uploads/symbotic-annual-report.pdf` and ingested by `uv run pytest tests/live/test_ingestion_live.py -m live -s`. An upload endpoint arrives with the API in Phase 8.

**One real brief:** with the servers running and the annual report ingested, open `notebooks/phase_04.ipynb` in VS Code, choose the project's `.venv` as the kernel, and run all cells. It takes about two minutes: it first re-ingests the report into a temporary folder to time it, then writes a brief from the web and the report (about 35 model requests, up to 6 searches), prints the LangSmith link to its trace, and writes `notebooks/phase_04_brief.md`.

**Live checks** against the real APIs and the running server (they spend free-tier quota):

```powershell
uv run pytest -m live -s
```

## Keeping secrets out of this repository

- API keys are read only from `.env`, and are never printed. The graph reads its settings in one place (`settings.py`); `search-mcp` reads only the search key, in its own `config.py`; `pdf-parser-mcp` needs no key.
- Your documents stay on your machine: `data/` (uploads and the vector store) is ignored by git, and document text is never sent to a web search.
- `nbstripout` removes notebook outputs at commit time, so nothing a cell printed is committed.
- A test (`tests/unit/test_no_secrets.py`) runs in CI and fails if any tracked file contains a key-shaped string, an actual key value, a tracked `.env`, or a notebook with outputs.

## Project layout

```text
settings.py              the graph's only reader of environment variables
app/
  ingestion.py           a PDF into stored chunks: read, chunk, embed, store
  pdf_parser_client.py   the client for pdf-parser-mcp
graph/
  llm.py                 model gateway: retries, limits, the run budget, embeddings
  mcp_client.py          McpToolbox: the graph's only way to reach web tools
  extraction.py          P-02 and the word-for-word checks
  document_store.py      FAISS indexes and chunks, one folder per document
  retrieval.py           keyword and meaning search, merged by rank
  run_budget.py          a run's request budget, sized to its plan
  schemas.py             what each model call must return
  state.py               what a run keeps between steps
  supervisor.py          builds the graph
  report.py              renders a brief as Markdown
  self_rag/              the Self-RAG subgraph: its state, nodes, and wiring
  prompts/               prompt text, one module per prompt group, each with a version
  nodes/                 planner, web researcher, document researcher, writer
mcp_servers/
  search_mcp/            web_search and fetch_page; its own settings; the address check
  pdf_parser_mcp/        reads PDF pages, only inside data/uploads/
scripts/
  start_mcp_servers.ps1  starts both servers and waits until they listen
  make_review_pack.py    builds the pack for the outside test review at each phase close
  make_test_inventory.py lists the tests each phase added, from the tests and git history
tests/
  unit/                  scripted tests: no keys, no network
  live/                  real API checks, run on request
notebooks/
  phase_04.ipynb         ingestion timed, one real run with the annual report, and the Phase 4 record
  phase_04_brief.md      the brief it wrote
  phase_03.ipynb         the Phase 3 record (re-running it now uses the Phase 4 graph)
  phase_03_brief.md      the Phase 3 brief on the same topic
  phase_02.ipynb         the Phase 2 record (re-running it now uses the Phase 4 graph)
  phase_02_brief.md      the Phase 2 brief on the same topic
  phase_01.ipynb         the Phase 1 record (no longer runs: its search client was removed)
  phase_01_brief.md      the Phase 1 brief on the same topic
data/                    your uploads and the vector store; ignored by git
```

## Roadmap

| Phase | Delivers | Status |
| --- | --- | --- |
| 1 | Planner, researcher, and writer produce a cited brief from a topic | Done |
| 2 | Web research through a custom MCP server; only passages copied word for word are kept | Done |
| 3 | One researcher per sub-question, in parallel, with a rate limit per model and a request budget per run | Done |
| 4 | Your own PDFs as a second source, through a Self-RAG subgraph | Done |
| 5 | Every claim checked against its cited passage; failed claims revised | Next |
| 6 | Human approval before release; runs survive a restart | Planned |
| 7 | Preferences that carry over between runs | Planned |
| 8 | REST API (FastAPI) | Planned |
| 9 | Background jobs and live progress | Planned |
| 10 | Metrics from traces, evaluation, a user trial | Planned |
| 11 | Docker, cloud deployment, a remote MCP server | Planned |
| 12 | Optional extras | Optional |
