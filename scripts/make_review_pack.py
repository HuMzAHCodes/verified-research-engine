"""Build the pack for the outside test review (D-60).

At every phase close, another AI tool reviews the project's code and tests to
find what the tests miss. This script builds what it is given:

    review_pack/vre-review-pack-<tag>-<commit>.zip
        REVIEW_PACK.md   what the project is, how to read the pack, the file map,
                         the coverage summary, every test with what it checks,
                         and the review prompt with its answer format
        ALL_FILES.md     every packed file in one document, for tools that
                         cannot open a zip of many files
        code/...         the packed files, at their paths in the repository

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
CODE_LANGUAGES = {".py": "python", ".ps1": "powershell", ".toml": "toml", ".yml": "yaml"}

REVIEW_PROMPT = """You are a senior test engineer reviewing a Python project's test suite. Your job
is to find what the tests miss, not to praise them. The code and every test are
in this pack (`code/`, or all in one document in `ALL_FILES.md`); this file maps
them, lists every test with what it checks, and shows which lines no scripted
test runs.

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
    module_is_live = "pytestmark = pytest.mark.live" in source
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
        "- `code/` holds the project's code and tests at their paths in the repository; "
        "`ALL_FILES.md` holds the same files in one document.",
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
        "| File | Lines | What it is |",
        "| --- | --- | --- |",
        *(
            f"| `{path}` | {line_count} | {file_summaries.get(path, '')} |"
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


def render_all_files(texts_by_path: dict[str, str]) -> str:
    sections = ["# All packed files", ""]
    for path, text in texts_by_path.items():
        language = CODE_LANGUAGES.get(Path(path).suffix, "")
        fence = "````" if "```" in text else "```"
        sections += [f"## `{path}`", "", f"{fence}{language}", text.rstrip("\n"), fence, ""]
    return "\n".join(sections)


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
    review_pack_text = render_review_pack(
        tag=tag,
        commit=commit,
        description=project_description(texts_by_path.get("README.md", "")),
        file_lines={path: len(text.splitlines()) for path, text in texts_by_path.items()},
        file_summaries={path: module_summary(path, text) for path, text in texts_by_path.items()},
        rows=coverage_rows(coverage_report),
        totals=coverage_report["totals"],
        inventory=inventory,
    )
    all_files_text = render_all_files(texts_by_path)

    leaks = secrets_found(
        {**texts_by_path, "REVIEW_PACK.md": review_pack_text, "ALL_FILES.md": all_files_text},
        real_key_values(),
    )
    if leaks:
        raise SystemExit("Not built: " + "; ".join(leaks))

    zip_path = PACK_FOLDER / f"vre-review-pack-{tag}-{commit}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as pack_zip:
        pack_zip.writestr("REVIEW_PACK.md", review_pack_text)
        pack_zip.writestr("ALL_FILES.md", all_files_text)
        for path, text in texts_by_path.items():
            pack_zip.writestr(f"code/{path}", text)
    (PACK_FOLDER / "REVIEW_PACK.md").write_text(review_pack_text, encoding="utf-8")

    test_count = sum(len(tests) for _, tests in inventory.values())
    print(f"Packed {len(paths)} files and {test_count} test functions")
    print(f"Coverage: {round(coverage_report['totals']['percent_covered'], 1)}%")
    print(f"Wrote {zip_path.relative_to(PROJECT_ROOT)} ({zip_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
