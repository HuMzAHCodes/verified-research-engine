"""Write the list of tests each phase added into the docs' testing folder.

The testing folder (docs repo, `testing/`) holds one file per phase. Each file
has a short part written by hand at the phase close (what the tests prove,
the planned test cases ticked, what broke), then a part this script writes:
every test function the phase added, grouped by file, with what it checks.

Run from the project root, at a phase close:

    uv run python scripts/make_test_inventory.py
    uv run python scripts/make_test_inventory.py --output <folder>

It reads the tests exactly as they are written; it asks nothing of them:

- what a test checks: its docstring's first paragraph, else the comment above
  its first statement (a leading "CONCEPT: ..." sentence left out, since it
  names the concept, not the check), else its name;
- live or scripted: the existing `live` marker;
- the phase that added it: the first commit that added the test to its file,
  from git history. A commit's phase is read from its message prefix
  (`[P2 mcp-client]` is Phase 2). A `[review]` commit is phase-close work
  (the outside test review, D-60, and this folder) and belongs to the close
  of the last phase tagged before it; any other commit, to the phase after the
  last tag before it.

Only the part below GENERATED_MARKER is replaced; the part above it is kept
as written. A file that does not exist yet is created with a placeholder
summary above the marker.
"""

import argparse
import ast
import dataclasses
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_review_pack import InventoryEntry, inventory_of_tests  # noqa: E402

DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "docs" / "testing"
TEST_FILE_PATTERN = re.compile(r"tests/.*test_[^/]*\.py$")
PHASE_TAG_PATTERN = re.compile(r"phase-(\d+)$")
PHASE_PREFIX_PATTERN = re.compile(r"^\[P(\d+)\b")
REVIEW_PREFIX = "[review]"
# Lines in `git log -p` output: a commit header written by our --format, the
# file a diff is about, and an added test function.
COMMIT_LINE_PATTERN = re.compile(r"^COMMIT ([0-9a-f]+) (.*)$")
DIFF_FILE_PATTERN = re.compile(r"^\+\+\+ b/(.+)$")
ADDED_TEST_PATTERN = re.compile(r"^\+\s*(?:async\s+)?def (test_\w+)\s*\(")
MAX_DESCRIPTION_CHARS = 300
CONCEPT_SENTENCE = re.compile(r"^CONCEPT:.*?(?:\.\s+|$)")
GENERATED_MARKER = "<!-- Below this line is written by scripts/make_test_inventory.py. -->"


@dataclass(frozen=True)
class CommitInfo:
    commit: str
    subject: str


@dataclass(frozen=True)
class OriginOfTest:
    """Where a test was first added: the phase, and whether at the phase close's review."""

    phase: int
    at_review: bool
    commit: str
    subject: str


@dataclass(frozen=True)
class InventoriedTest:
    path: str
    entry: InventoryEntry
    origin: OriginOfTest | None  # None: not yet committed


def shortened(text: str, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """The text on one line, cut at a word to at most max_chars, with "..." if cut."""
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars].rsplit(" ", 1)[0] + " ..."


def leading_comment_block(source_lines: list[str], function_node: ast.AST) -> str:
    """Every comment line above a test's first statement, joined, the CONCEPT sentence left out."""
    first_statement_line = function_node.body[0].lineno
    comment_lines = [
        source_line.strip().lstrip("#").strip()
        for source_line in source_lines[function_node.lineno : first_statement_line - 1]
        if source_line.strip().startswith("#")
    ]
    return CONCEPT_SENTENCE.sub("", " ".join(comment_lines)).strip()


def descriptions_of_tests(source: str) -> dict[str, str]:
    """What each test function checks, read from how it is written; by test name."""
    module = ast.parse(source)
    source_lines = source.splitlines()
    descriptions = {}
    for node in module.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        docstring = ast.get_docstring(node) or ""
        first_paragraph = docstring.strip().split("\n\n")[0]
        description = (
            first_paragraph
            or leading_comment_block(source_lines, node)
            or node.name.removeprefix("test_").replace("_", " ")
        )
        descriptions[node.name] = shortened(description)
    return descriptions


def git_output(*arguments: str) -> str:
    """Git's output as text. Read as UTF-8, whatever the system's code page: the tests
    hold characters such as "\ufb01" and "\u00b2", which Windows' default code page
    cannot read."""
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout


def first_added_commits(
    log_text: str,
) -> tuple[dict[tuple[str, str], CommitInfo], dict[str, CommitInfo]]:
    """The first commit adding each test, by (file, name), and by name alone.

    log_text is `git log --reverse -p` output, oldest commit first, with each
    commit's header written as "COMMIT <full hash> <subject>". By name alone is the
    fallback for a test whose file was later renamed.
    """
    by_file_and_name: dict[tuple[str, str], CommitInfo] = {}
    by_name: dict[str, CommitInfo] = {}
    current_commit: CommitInfo | None = None
    current_path = ""
    for line in log_text.splitlines():
        commit_match = COMMIT_LINE_PATTERN.match(line)
        if commit_match:
            current_commit = CommitInfo(commit_match[1], commit_match[2])
            continue
        file_match = DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_path = file_match[1]
            continue
        test_match = ADDED_TEST_PATTERN.match(line)
        if test_match and current_commit is not None:
            by_file_and_name.setdefault((current_path, test_match[1]), current_commit)
            by_name.setdefault(test_match[1], current_commit)
    return by_file_and_name, by_name


def last_phase_tag_before(commit_position: int, tag_positions: dict[int, int]) -> int:
    """The highest phase number whose tag comes at or before this commit; 0 if none."""
    return max(
        (phase for phase, position in tag_positions.items() if position <= commit_position),
        default=0,
    )


def origin_of(
    commit_info: CommitInfo, commit_position: int, tag_positions: dict[int, int]
) -> OriginOfTest:
    """The phase a commit belongs to, from its message prefix, else from the tags."""
    prefix_match = PHASE_PREFIX_PATTERN.match(commit_info.subject)
    if prefix_match:
        phase, at_review = int(prefix_match[1]), False
    elif commit_info.subject.startswith(REVIEW_PREFIX):
        phase, at_review = last_phase_tag_before(commit_position, tag_positions), True
    else:
        phase, at_review = last_phase_tag_before(commit_position, tag_positions) + 1, False
    # Phase 0 was the set-up before Phase 1; its tests are listed with Phase 1.
    return OriginOfTest(max(phase, 1), at_review, commit_info.commit, commit_info.subject)


def phase_heading(phase: int, at_review: bool) -> str:
    if at_review:
        return f"Added at the Phase {phase} close"
    return f"Added while building Phase {phase}"


def render_generated_part(phase: int, tests: list[InventoriedTest], head_commit: str) -> str:
    """The generated part of one phase's file: its tests, grouped by when and by file."""
    scripted_count = sum(1 for test in tests if not test.entry.live)
    live_count = len(tests) - scripted_count
    lines = [
        GENERATED_MARKER,
        "",
        f"## Tests added in Phase {phase}",
        "",
        f"{len(tests)} test functions first added in this phase: {scripted_count} scripted, "
        f"{live_count} live (a parametrized test counts once). Read from the tests and git "
        f"history at commit `{head_commit}`; a test changed in a later phase is listed "
        "here, where it was added, as it reads now.",
        "",
    ]
    for at_review in (False, True):
        group = [
            test
            for test in tests
            if (test.origin is not None) and test.origin.at_review == at_review
        ]
        if not group:
            continue
        lines += [f"### {phase_heading(phase, at_review)}", ""]
        for path in sorted({test.path for test in group}):
            lines += [f"#### `{path}`", ""]
            for test in (test for test in group if test.path == path):
                lines.append(f"- {describe(test.entry)}")
            lines.append("")
    not_committed = [test for test in tests if test.origin is None]
    if not_committed:
        lines += ["### Not yet committed", ""]
        lines += [f"- `{test.path}`: {describe(test.entry)}" for test in not_committed]
        lines.append("")
    return "\n".join(lines)


def describe(entry: InventoryEntry) -> str:
    labels = ", ".join(
        label
        for label, present in (("live", entry.live), ("parametrized", entry.parametrized))
        if present
    )
    return f"`{entry.name}`{f' ({labels})' if labels else ''}: {entry.description}"


def placeholder_summary(phase: int) -> str:
    return (
        f"# Phase {phase}: testing\n\n"
        "(The summary is written by hand at the phase close: what the tests prove, the "
        "kinds of tests used, the planned test cases ticked, what broke, and the suite's "
        "size and coverage at the close.)\n\n"
    )


def merge_into_file(existing_text: str | None, phase: int, generated_part: str) -> str:
    """The file's hand-written part kept as it is, the generated part replaced."""
    if existing_text is None:
        hand_written = placeholder_summary(phase)
    elif GENERATED_MARKER in existing_text:
        hand_written = existing_text.split(GENERATED_MARKER)[0]
    else:
        hand_written = existing_text.rstrip("\n") + "\n\n"
    return hand_written + generated_part.rstrip("\n") + "\n"


def collect_tests() -> tuple[list[InventoriedTest], str]:
    """Every test function in the tracked test files, with the phase that added it."""
    test_paths = [
        path
        for path in git_output("ls-files", "tests").splitlines()
        if TEST_FILE_PATTERN.match(path)
    ]
    commit_order = git_output("rev-list", "--reverse", "HEAD").split()
    position_of = {commit: position for position, commit in enumerate(commit_order)}
    tag_positions: dict[int, int] = {}
    for tag_line in git_output(
        "tag", "--list", "phase-*", "--format=%(refname:short) %(objectname)"
    ).splitlines():
        tag_name, tag_commit = tag_line.split()
        dereferenced = git_output("rev-list", "-n", "1", tag_commit).strip()
        tag_match = PHASE_TAG_PATTERN.match(tag_name)
        if tag_match and dereferenced in position_of:
            tag_positions[int(tag_match[1])] = position_of[dereferenced]

    log_text = git_output(
        "log", "--reverse", "-p", "--no-color", "--format=COMMIT %H %s", "--", "tests"
    )
    by_file_and_name, by_name = first_added_commits(log_text)

    inventoried: list[InventoriedTest] = []
    for path in test_paths:
        source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        _, entries = inventory_of_tests(source)  # names, live, parametrized (D-60's reading)
        descriptions = descriptions_of_tests(source)
        for listed_entry in entries:
            entry = dataclasses.replace(
                listed_entry,
                description=descriptions.get(listed_entry.name, listed_entry.description),
            )
            commit_info = by_file_and_name.get((path, entry.name)) or by_name.get(entry.name)
            origin = None
            if commit_info is not None:
                origin = origin_of(commit_info, position_of[commit_info.commit], tag_positions)
            inventoried.append(InventoriedTest(path, entry, origin))
    return inventoried, git_output("rev-parse", "--short", "HEAD").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    output_folder: Path = parser.parse_args().output
    if not output_folder.parent.is_dir():
        raise SystemExit(f"{output_folder.parent} does not exist; is the docs repo in place?")
    output_folder.mkdir(exist_ok=True)

    inventoried, head_commit = collect_tests()
    current_phase = max(
        (test.origin.phase for test in inventoried if test.origin is not None), default=1
    )
    phases = sorted(
        {test.origin.phase if test.origin is not None else current_phase for test in inventoried}
    )
    for phase in phases:
        phase_tests = [
            test
            for test in inventoried
            if (test.origin.phase if test.origin is not None else current_phase) == phase
        ]
        phase_path = output_folder / f"phase-{phase:02d}.md"
        existing_text = phase_path.read_text(encoding="utf-8") if phase_path.exists() else None
        phase_path.write_text(
            merge_into_file(
                existing_text, phase, render_generated_part(phase, phase_tests, head_commit)
            ),
            encoding="utf-8",
        )
        scripted = sum(1 for test in phase_tests if not test.entry.live)
        print(f"{phase_path.name}: {len(phase_tests)} tests ({scripted} scripted)")
    print(f"{len(inventoried)} test functions in all, at commit {head_commit}")


if __name__ == "__main__":
    main()
