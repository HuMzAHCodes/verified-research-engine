"""Build the pack for the outside test review (D-60).

At every phase close, another AI tool reviews the project's code and tests to
find what the tests miss. This script builds what it is given:

    review_pack/vre-review-pack-<tag>-<commit>.zip
        REVIEW_PACK.md   what the project is, how to read the pack, the file map,
                         the coverage summary, every test with what it checks,
                         and the review prompt with its answer format
        FILES_<n>_<area>.md
                         the packed files in a few documents, one per area of
                         the project, each holding an area's code and its tests
                         (for tools that cannot open a zip of many files)
        code/...         the packed files, at their paths in the repository

REVIEW_PACK.md and the FILES documents are also written next to the zip, ready
to upload. Each FILES document stays under PART_LIMIT_BYTES: the first tool
used (Gemini, at the Phase 4 review) stopped reading one long document at about
349 KB, partway through, without saying so. Each document therefore ends with a
line that names it and counts its files, so a tool can be asked to quote it.

Run from the project root:

    uv run python scripts/make_review_pack.py

What goes in: only files git tracks, under graph/, app/, mcp_servers/,
tests/, scripts/, and .github/, plus settings.py, pyproject.toml, README.md,
and .env.example. What never goes in: .env, data/, notebooks, the lock file,
and the private design documents (they are not in this repository). Before
anything is written, every packed text is checked for the real key values in
this environment and for key-shaped strings, with the same rules as
tests/unit/test_no_secrets.py; the script stops if it finds one.

The coverage numbers come from running the scripted tests (`-m "not live"`)
with coverage, here, so the pack always describes the code it contains.
"""

import ast
import fnmatch
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.unit.test_no_secrets import KEY_SHAPES, real_key_values  # noqa: E402

PACK_FOLDER = PROJECT_ROOT / "review_pack"
COVERAGE_JSON_PATH = PACK_FOLDER / "coverage.json"
INCLUDED_FOLDERS = ("graph/", "app/", "mcp_servers/", "tests/", "scripts/", ".github/")
INCLUDED_FILES = frozenset({"settings.py", "pyproject.toml", "README.md", ".env.example"})
# Never packed, even if one day tracked by mistake.
EXCLUDED_NAMES = frozenset({".env"})
NON_PYTHON_SUMMARIES = {
    ".env.example": "Every setting the project reads; secrets are left empty.",
    ".github/workflows/ci.yml": "CI: ruff and the scripted tests on every push, placeholder keys.",
    "README.md": "What the project does, how to set it up and run it.",
    "pyproject.toml": "Dependencies, and the settings of pytest, ruff, and coverage.",
    "scripts/start_mcp_servers.ps1": "Starts the local MCP servers on Windows and waits for them.",
}
# A part stays well under the 349 KB where the first review tool stopped reading.
PART_LIMIT_BYTES = 250_000
# Test helpers every area's tests use: repeated in every part.
SHARED_PATHS = ("tests/conftest.py", "tests/pdf_fixtures.py")


@dataclass(frozen=True)
class ReviewArea:
    slug: str
    title: str
    patterns: tuple[str, ...]


# A file goes to the first area one of whose patterns it matches; the last
# area takes everything else. Code sits in the same part as its tests.
REVIEW_AREAS = (
    ReviewArea(
        "web",
        "web search and page reading: search-mcp, the MCP client, web research",
        (
            "mcp_servers/search_mcp/*",
            "graph/mcp_client.py",
            "graph/extraction.py",
            "graph/nodes/researcher.py",
            "graph/prompts/web_findings.py",
            "tests/unit/test_url_safety.py",
            "tests/unit/test_fetch.py",
            "tests/unit/test_search.py",
            "tests/unit/test_search_mcp_server.py",
            "tests/unit/test_mcp_client.py",
            "tests/unit/test_fake_toolbox.py",
            "tests/unit/test_extraction.py",
            "tests/unit/test_web_findings_prompt.py",
            "tests/unit/test_researcher.py",
            "tests/unit/test_graph_imports.py",
            "tests/live/test_search_mcp_live.py",
            "tests/live/test_mcp_client_live.py",
            "tests/live/test_extraction_live.py",
        ),
    ),
    ReviewArea(
        "documents",
        "documents: pdf-parser-mcp, ingestion, the store, retrieval, Self-RAG",
        (
            "mcp_servers/pdf_parser_mcp/*",
            "app/*",
            "graph/document_store.py",
            "graph/retrieval.py",
            "graph/self_rag/*",
            "graph/prompts/self_rag.py",
            "graph/nodes/document_researcher.py",
            "tests/unit/test_pdf_*.py",
            "tests/unit/test_page_store.py",
            "tests/unit/test_app_imports.py",
            "tests/unit/test_chunking.py",
            "tests/unit/test_ingestion.py",
            "tests/unit/test_document_store.py",
            "tests/unit/test_retrieval.py",
            "tests/unit/test_self_rag*.py",
            "tests/unit/test_document_researcher.py",
            "tests/live/test_pdf_parser_live.py",
            "tests/live/test_ingestion_live.py",
            "tests/live/test_retrieval_live.py",
            "tests/live/test_self_rag_live.py",
        ),
    ),
    ReviewArea(
        "graph",
        "the research graph: planner, writer, supervisor, state, run budget, the brief",
        (
            "graph/nodes/planner.py",
            "graph/nodes/writer.py",
            "graph/supervisor.py",
            "graph/state.py",
            "graph/schemas.py",
            "graph/report.py",
            "graph/run_budget.py",
            "graph/prompts/planner.py",
            "graph/prompts/writer.py",
            "tests/unit/test_graph.py",
            "tests/unit/test_graph_documents.py",
            "tests/unit/test_planner.py",
            "tests/unit/test_writer.py",
            "tests/unit/test_report.py",
            "tests/unit/test_run_budget.py",
            "tests/live/test_graph_live.py",
            "tests/live/test_graph_documents_live.py",
            "tests/live/test_planner_live.py",
        ),
    ),
    ReviewArea(
        "core",
        "core: settings, the model gateway and its limits, secrets, tooling",
        ("*",),
    ),
)
CODE_LANGUAGES = {".py": "python", ".ps1": "powershell", ".toml": "toml", ".yml": "yaml"}

REVIEW_PROMPT = """You are a senior test engineer reviewing a Python project's test suite. Your job
is to find what the tests miss, not to praise them. The code and every test are
in this pack (the FILES documents, or `code/` in the zip); this file maps
them, lists every test with what it checks, and shows which lines no scripted
test runs. If you were given only some of the FILES documents, review only the
areas whose code you can read, say which those are, and make no finding about
code you have not read.

Find, in order of importance:

1. Behaviours of the code that no test checks, or checks only on the easy path.
2. Tests that cannot fail, or whose assertions are too weak to catch a wrong result.
3. Possible bugs you can see in the code itself, with the input that shows them.
4. Tests that could be flaky: timing, ordering of parallel work, shared state.
5. Security gaps a test should pin down: the private-address check on URLs
   (redirects, unusual address forms), paths outside the upload folder,
   text from documents or web pages posing as instructions or as tags,
   secrets reaching logs or errors.

Look in particular at limits and failure paths: the request budget, the rate
limiters and retries, a tool server that is down or answers badly, empty and
very large inputs, Unicode in text that is matched word for word, and work that
runs in parallel branches. The coverage table shows lines no scripted test runs;
start there, but most gaps are in lines that run without being checked.

Rules for your suggestions:

- Scripted tests only: no network, no real API keys, deterministic. Use the
  existing helpers in `tests/conftest.py` (ScriptedModel, ScriptedEmbeddings,
  FakeToolbox, make_gateway) and `tests/pdf_fixtures.py` where they fit.
- Do not suggest testing third-party libraries themselves.
- Do not suggest redesigns. If a finding needs a code change, say what fails and why.
- Say which existing test comes closest, if any, and why it does not catch the case.
- At most 25 findings, most important first. Quality over quantity: a finding
  you are unsure of is marked as such.

Answer in exactly this format, one block per finding:

### F-<number>: <short title>
- Kind: missing test | test that cannot fail | weak assertion | possible bug | flaky | security
- Where: <file> :: <function or behaviour>
- Scenario: <the input and state>
- Expected: <what should happen>
- Why the current tests miss it: <the closest test's name and the gap, or "no test touches it">
- Severity: high | medium | low
- Confidence: high | medium | low
- Test sketch:
```python
# a pytest test, using the project's helpers
```

End with a short list, "Areas that look well tested", so the gaps can be told
apart from areas you did not reach."""


@dataclass(frozen=True)
class CoverageRow:
    path: str
    statements: int
    missed: int
    branches: int
    partial_branches: int
    percent: float
    missing: str


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    description: str
    live: bool
    parametrized: bool


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def pack_paths(tracked_paths: list[str]) -> list[str]:
    """The tracked files that go in the pack, sorted."""
    return sorted(
        path
        for path in tracked_paths
        if Path(path).name not in EXCLUDED_NAMES
        and (path in INCLUDED_FILES or path.startswith(INCLUDED_FOLDERS))
    )


def line_ranges(line_numbers: list[int]) -> str:
    """[3, 4, 5, 9] -> "3-5, 9"."""
    ranges: list[str] = []
    for line_number in sorted(line_numbers):
        if ranges and ranges[-1].split("-")[-1] == str(line_number - 1):
            ranges[-1] = f"{ranges[-1].split('-')[0]}-{line_number}"
        else:
            ranges.append(str(line_number))
    return ", ".join(ranges)


def coverage_rows(coverage_report: dict) -> list[CoverageRow]:
    """One row per measured file, from coverage.py's JSON report."""
    rows = []
    for path, file_report in coverage_report["files"].items():
        summary = file_report["summary"]
        rows.append(
            CoverageRow(
                path=path.replace("\\", "/"),
                statements=summary["num_statements"],
                missed=summary["missing_lines"],
                branches=summary.get("num_branches", 0),
                partial_branches=summary.get("num_partial_branches", 0),
                percent=round(summary["percent_covered"], 1),
                missing=line_ranges(file_report.get("missing_lines", [])),
            )
        )
    return sorted(rows, key=lambda row: row.path)


def first_line(text: str | None) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def leading_comment(source_lines: list[str], function_node: ast.AST) -> str:
    """A comment before a test's first statement, which says what the whole test checks.

    Comments further down describe one step, so they are not used; nor are
    CONCEPT lines, which name the concept rather than the check.
    """
    first_statement_line = function_node.body[0].lineno
    for source_line in source_lines[function_node.lineno : first_statement_line - 1]:
        stripped = source_line.strip()
        if stripped.startswith("#") and not stripped.lstrip("# ").startswith("CONCEPT:"):
            return stripped.lstrip("#").strip()
    return ""


def inventory_of_tests(source: str) -> tuple[str, list[InventoryEntry]]:
    """A test module's summary line and its tests, each with what it checks."""
    module = ast.parse(source)
    source_lines = source.splitlines()
    # Only a real module-level `pytestmark = ...live...` marks the module, not the
    # same words inside a string (this script's own tests hold them in one).
    module_is_live = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        )
        and "mark.live" in ast.unparse(node.value)
        for node in module.body
    )
    entries = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        description = (
            first_line(ast.get_docstring(node))
            or leading_comment(source_lines, node)
            or node.name.removeprefix("test_").replace("_", " ")
        )
        entries.append(
            InventoryEntry(
                name=node.name,
                description=description,
                live=module_is_live or any("mark.live" in d for d in decorators),
                parametrized=any("parametrize" in d for d in decorators),
            )
        )
    return first_line(ast.get_docstring(module)), entries


def module_summary(path: str, source: str) -> str:
    if path in NON_PYTHON_SUMMARIES:
        return NON_PYTHON_SUMMARIES[path]
    if not path.endswith(".py"):
        return ""
    try:
        return first_line(ast.get_docstring(ast.parse(source)))
    except SyntaxError:
        return ""


def secrets_found(texts_by_name: dict[str, str], key_values: list[str]) -> list[str]:
    """Where a real key value or a key-shaped string occurs: "<name>: <what>"."""
    found = []
    for name, text in texts_by_name.items():
        if any(key_value in text for key_value in key_values):
            found.append(f"{name}: a real key value")
        for shape_name, pattern in KEY_SHAPES.items():
            if pattern.search(text):
                found.append(f"{name}: {shape_name}")
    return found


def project_description(readme_text: str) -> str:
    """The README's first paragraph that is not a heading or a quote."""
    for paragraph in readme_text.split("\n\n"):
        stripped = paragraph.strip()
        if stripped and not stripped.startswith(("#", ">", "|", "```")):
            return stripped
    return ""


def render_review_pack(
    *,
    tag: str,
    commit: str,
    description: str,
    file_lines: dict[str, int],
    file_summaries: dict[str, str],
    rows: list[CoverageRow],
    totals: dict,
    inventory: dict[str, tuple[str, list[InventoryEntry]]],
    part_names: list[str],
    part_by_path: dict[str, str],
) -> str:
    scripted_count = sum(1 for _, tests in inventory.values() for t in tests if not t.live)
    live_count = sum(1 for _, tests in inventory.values() for t in tests if t.live)
    lines = [
        f"# Review pack: Verified Research Engine at `{tag}` ({commit})",
        "",
        description,
        "",
        "## How to read this pack",
        "",
        "- The code and tests come in "
        + ", ".join(f"`{name}`" for name in part_names)
        + ": one document per area of the project, each with that area's code and its "
        "tests; the zip also holds them in `code/`, at their paths in the repository. "
        "The Part column of the file map says where each file is.",
        "- Tests marked **live** call real APIs and run only on request; every other test is "
        "**scripted**: no network, no keys, run on every push "
        '(`uv run pytest -m "not live"`).',
        "- Test doubles live in `tests/conftest.py`; small PDFs are built in "
        "`tests/pdf_fixtures.py`.",
        f"- {len(file_lines)} files; {scripted_count} scripted and {live_count} live test "
        "functions (a parametrized test counts once).",
        "",
        "## The review request",
        "",
        REVIEW_PROMPT,
        "",
        "## Coverage of the scripted tests",
        "",
        f"Lines: {totals['covered_lines']} of {totals['num_statements']} run; branches: "
        f"{totals.get('covered_branches', 0)} of {totals.get('num_branches', 0)} taken; "
        f"overall {round(totals['percent_covered'], 1)}%. Lines are listed where no "
        "scripted test runs them.",
        "",
        "| File | Statements | Missed | Branches | Partial | Cover | Lines not run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| `{row.path}` | {row.statements} | {row.missed} | {row.branches} | "
            f"{row.partial_branches} | {row.percent}% | {row.missing or '-'} |"
            for row in rows
        ),
        "",
        "## Files",
        "",
        "| File | Part | Lines | What it is |",
        "| --- | --- | --- | --- |",
        *(
            f"| `{path}` | {part_by_path.get(path, '')} | {line_count} | "
            f"{file_summaries.get(path, '')} |"
            for path, line_count in file_lines.items()
        ),
        "",
        "## Every test",
        "",
    ]
    for test_path, (module_line, tests) in inventory.items():
        lines.append(f"### `{test_path}`")
        lines.append("")
        if module_line:
            lines.append(module_line)
            lines.append("")
        for test in tests:
            labels = ", ".join(
                label
                for label, present in (("live", test.live), ("parametrized", test.parametrized))
                if present
            )
            suffix = f" ({labels})" if labels else ""
            lines.append(f"- `{test.name}`{suffix}: {test.description}")
        lines.append("")
    return "\n".join(lines)


def area_of(path: str) -> str:
    """The slug of the area a packed file belongs to."""
    for area in REVIEW_AREAS:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in area.patterns):
            return area.slug
    return REVIEW_AREAS[-1].slug


def split_into_parts(paths: list[str]) -> dict[str, list[str]]:
    """Each area's files, in area order; the shared test helpers in every part."""
    shared = [path for path in paths if path in SHARED_PATHS]
    parts: dict[str, list[str]] = {area.slug: list(shared) for area in REVIEW_AREAS}
    for path in paths:
        if path not in SHARED_PATHS:
            parts[area_of(path)].append(path)
    return {slug: part_paths for slug, part_paths in parts.items() if len(part_paths) > len(shared)}


def part_file_name(number: int, slug: str) -> str:
    return f"FILES_{number}_{slug}.md"


def part_end_line(number: int, part_count: int, file_count: int) -> str:
    return f"End of part {number} of {part_count}: {file_count} files."


def render_part(texts_by_path: dict[str, str], *, number: int, part_count: int, title: str) -> str:
    sections = [
        f"# Review files, part {number} of {part_count}: {title}",
        "",
        f"{len(texts_by_path)} files. The shared test helpers (tests/conftest.py, "
        "tests/pdf_fixtures.py) are repeated in every part.",
        "",
    ]
    for path, text in texts_by_path.items():
        language = CODE_LANGUAGES.get(Path(path).suffix, "")
        fence = "````" if "```" in text else "```"
        sections += [f"## `{path}`", "", f"{fence}{language}", text.rstrip("\n"), fence, ""]
    sections.append(part_end_line(number, part_count, len(texts_by_path)))
    return "\n".join(sections) + "\n"


def render_parts(texts_by_path: dict[str, str]) -> dict[str, str]:
    """{file name: text} for every part; stops if a part is over PART_LIMIT_BYTES."""
    parts = split_into_parts(list(texts_by_path))
    titles = {area.slug: area.title for area in REVIEW_AREAS}
    rendered = {}
    for number, (slug, part_paths) in enumerate(parts.items(), start=1):
        name = part_file_name(number, slug)
        rendered[name] = render_part(
            {path: texts_by_path[path] for path in part_paths},
            number=number,
            part_count=len(parts),
            title=titles[slug],
        )
        size = len(rendered[name].encode("utf-8"))
        if size > PART_LIMIT_BYTES:
            raise SystemExit(
                f"{name} is {size // 1024} KB, over the {PART_LIMIT_BYTES // 1024} KB a part "
                "may hold; split its area in REVIEW_AREAS."
            )
    return rendered


def run_scripted_tests_with_coverage() -> dict:
    PACK_FOLDER.mkdir(exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not live",
            "-p",
            "no:cacheprovider",
            "--cov",
            f"--cov-report=json:{COVERAGE_JSON_PATH}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "")
    if completed.returncode != 0:
        print(completed.stdout[-3000:])
        raise SystemExit("The scripted tests did not all pass; no pack was built.")
    return json.loads(COVERAGE_JSON_PATH.read_text(encoding="utf-8"))


def main() -> None:
    tag = git_output("describe", "--tags", "--abbrev=0")
    commit = git_output("rev-parse", "--short", "HEAD")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        # The pack holds the files as they are on disk; say they differ from the commit.
        commit += "-modified"
        print("Note: tracked files have uncommitted changes; the pack holds them as they are")
    print(f"Building the review pack for {tag} ({commit}); running the scripted tests first")
    coverage_report = run_scripted_tests_with_coverage()

    paths = pack_paths(git_output("ls-files").splitlines())
    texts_by_path = {
        path: (PROJECT_ROOT / path).read_text(encoding="utf-8", errors="replace") for path in paths
    }
    inventory = {
        path: inventory_of_tests(text)
        for path, text in texts_by_path.items()
        if re.match(r"tests/.*test_[^/]*\.py$", path)
    }
    parts = render_parts(texts_by_path)
    part_numbers_by_path: dict[str, list[str]] = {}
    for number, part_paths in enumerate(split_into_parts(paths).values(), start=1):
        for path in part_paths:
            part_numbers_by_path.setdefault(path, []).append(str(number))
    part_by_path = {path: ", ".join(numbers) for path, numbers in part_numbers_by_path.items()}
    review_pack_text = render_review_pack(
        tag=tag,
        commit=commit,
        description=project_description(texts_by_path.get("README.md", "")),
        file_lines={path: len(text.splitlines()) for path, text in texts_by_path.items()},
        file_summaries={path: module_summary(path, text) for path, text in texts_by_path.items()},
        rows=coverage_rows(coverage_report),
        totals=coverage_report["totals"],
        inventory=inventory,
        part_names=list(parts),
        part_by_path=part_by_path,
    )

    leaks = secrets_found(
        {**texts_by_path, **parts, "REVIEW_PACK.md": review_pack_text},
        real_key_values(),
    )
    if leaks:
        raise SystemExit("Not built: " + "; ".join(leaks))

    zip_path = PACK_FOLDER / f"vre-review-pack-{tag}-{commit}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as pack_zip:
        pack_zip.writestr("REVIEW_PACK.md", review_pack_text)
        for name, part_text in parts.items():
            pack_zip.writestr(name, part_text)
        for path, text in texts_by_path.items():
            pack_zip.writestr(f"code/{path}", text)
    for stale_part in PACK_FOLDER.glob("FILES_*.md"):
        stale_part.unlink()
    (PACK_FOLDER / "REVIEW_PACK.md").write_text(review_pack_text, encoding="utf-8")
    for name, part_text in parts.items():
        (PACK_FOLDER / name).write_text(part_text, encoding="utf-8")

    test_count = sum(len(tests) for _, tests in inventory.values())
    print(f"Packed {len(paths)} files and {test_count} test functions")
    print(f"Coverage: {round(coverage_report['totals']['percent_covered'], 1)}%")
    print(f"Wrote {zip_path.relative_to(PROJECT_ROOT)} ({zip_path.stat().st_size // 1024} KB)")
    for name, part_text in parts.items():
        print(
            f"  {name}: {len(part_text.encode('utf-8')) // 1024} KB, ends with: "
            f"{part_text.rstrip().splitlines()[-1]}"
        )


if __name__ == "__main__":
    main()
