"""The review pack builder's pure parts (D-60, the outside test review).

CONCEPT: outside-test-review. scripts/make_review_pack.py runs the scripted
tests and git, which these tests leave alone; they check what decides the
pack's contents and its safety: which files go in, how each test is listed,
how the coverage table reads, and that a key stops the build.
"""

import pytest

from scripts.make_review_pack import (
    INCLUDED_FILES,
    PART_LIMIT_BYTES,
    SHARED_PATHS,
    area_of,
    coverage_rows,
    inventory_of_tests,
    line_ranges,
    module_summary,
    pack_paths,
    project_description,
    render_part,
    render_parts,
    secrets_found,
    split_into_parts,
)

# Built at run time so this file never holds a key-shaped string (TC-NFR-09).
FAKE_TAVILY_KEY = "tvly-" + "a1" * 12
FAKE_UNSHAPED_KEY = "q" * 32


def test_only_public_code_and_tests_go_in_the_pack():
    tracked = [
        "graph/llm.py",
        "app/ingestion.py",
        "mcp_servers/search_mcp/url_safety.py",
        "tests/conftest.py",
        "scripts/start_mcp_servers.ps1",
        ".github/workflows/ci.yml",
        "settings.py",
        "pyproject.toml",
        "README.md",
        ".env.example",
        "notebooks/phase_04.ipynb",
        "notebooks/phase_04_brief.md",
        "uv.lock",
        ".gitignore",
        ".python-version",
        "data/uploads/report.pdf",
        "graph/.env",
        "graphs_elsewhere/llm.py",
    ]

    packed = pack_paths(tracked)

    assert packed == sorted(packed)
    assert "notebooks/phase_04.ipynb" not in packed
    assert "uv.lock" not in packed
    assert "data/uploads/report.pdf" not in packed
    # A .env is never packed, wherever it sits.
    assert "graph/.env" not in packed
    # Folder matching is by whole folder name, not by prefix of a name.
    assert "graphs_elsewhere/llm.py" not in packed
    assert set(packed) == {
        "graph/llm.py",
        "app/ingestion.py",
        "mcp_servers/search_mcp/url_safety.py",
        "tests/conftest.py",
        "scripts/start_mcp_servers.ps1",
        ".github/workflows/ci.yml",
        *INCLUDED_FILES,
    }


def test_missing_lines_are_shown_as_ranges():
    assert line_ranges([]) == ""
    assert line_ranges([7]) == "7"
    assert line_ranges([9, 3, 4, 5]) == "3-5, 9"
    assert line_ranges([36, 37, 38, 42, 43]) == "36-38, 42-43"
    assert line_ranges([10, 11, 13, 14, 15, 17]) == "10-11, 13-15, 17"


def test_coverage_rows_read_the_json_report_with_forward_slashes():
    report = {
        "files": {
            "mcp_servers\\search_mcp\\config.py": {
                "summary": {
                    "num_statements": 18,
                    "missing_lines": 4,
                    "num_branches": 2,
                    "num_partial_branches": 0,
                    "percent_covered": 70.0,
                },
                "missing_lines": [36, 37, 38, 42],
            },
            "app\\ingestion.py": {
                "summary": {
                    "num_statements": 124,
                    "missing_lines": 0,
                    "percent_covered": 100.0,
                },
                "missing_lines": [],
            },
        }
    }

    rows = coverage_rows(report)

    assert [row.path for row in rows] == ["app/ingestion.py", "mcp_servers/search_mcp/config.py"]
    config_row = rows[1]
    assert (config_row.statements, config_row.missed, config_row.branches) == (18, 4, 2)
    assert config_row.percent == 70.0
    assert config_row.missing == "36-38, 42"
    # A report without branch numbers (run without --cov-branch) still reads.
    assert (rows[0].branches, rows[0].partial_branches, rows[0].missing) == (0, 0, "")


TEST_MODULE_SOURCE = '''"""Checks for the widget.

More detail that the inventory leaves out.
"""

import pytest


def helper_not_a_test():
    pass


def test_with_a_docstring():
    """The widget turns blue.

    Second line left out.
    """
    assert True


def test_with_a_comment():
    # CONCEPT: some-concept. Names the concept, not the check.
    # The comment says what is checked.
    value = 1
    assert value


def test_with_a_comment_further_down():
    value = 1
    # Describes one step only.
    assert value


async def test_async_with_neither():
    assert True


@pytest.mark.live
def test_marked_live():
    assert True


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized(value):
    assert value


class TestNotListed:
    def test_inside_a_class(self):
        assert True
'''


def test_the_inventory_lists_each_test_with_what_it_checks():
    module_line, entries = inventory_of_tests(TEST_MODULE_SOURCE)

    assert module_line == "Checks for the widget."
    by_name = {entry.name: entry for entry in entries}
    assert list(by_name) == [
        "test_with_a_docstring",
        "test_with_a_comment",
        "test_with_a_comment_further_down",
        "test_async_with_neither",
        "test_marked_live",
        "test_parametrized",
    ]
    assert by_name["test_with_a_docstring"].description == "The widget turns blue."
    assert by_name["test_with_a_comment"].description == "The comment says what is checked."
    # A comment after the first statement describes one step, so the name is used.
    assert by_name["test_with_a_comment_further_down"].description == (
        "with a comment further down"
    )
    # With neither, the name itself is the description.
    assert by_name["test_async_with_neither"].description == "async with neither"
    assert by_name["test_marked_live"].live and not by_name["test_with_a_docstring"].live
    assert by_name["test_parametrized"].parametrized
    assert not by_name["test_marked_live"].parametrized


def test_a_module_marked_live_marks_every_test_live():
    source = "import pytest\n\npytestmark = pytest.mark.live\n\n\ndef test_one():\n    pass\n"

    module_line, entries = inventory_of_tests(source)

    assert module_line == ""
    assert [entry.live for entry in entries] == [True]


def test_the_live_marker_inside_a_string_does_not_mark_the_module():
    source = 'EXAMPLE = "pytestmark = pytest.mark.live"\n\n\ndef test_one():\n    pass\n'

    _, entries = inventory_of_tests(source)

    assert [entry.live for entry in entries] == [False]


def test_a_list_of_marks_with_live_marks_the_module():
    source = (
        "import pytest\n\npytestmark = [pytest.mark.live, pytest.mark.slow]\n\n\n"
        "def test_one():\n    pass\n"
    )

    _, entries = inventory_of_tests(source)

    assert [entry.live for entry in entries] == [True]


def test_a_key_shaped_string_or_a_real_key_value_stops_the_build():
    texts = {
        "graph/clean.py": "print('nothing secret here')",
        "graph/shaped.py": f"TOKEN = '{FAKE_TAVILY_KEY}'",
        "REVIEW_PACK.md": f"a key with no shape: {FAKE_UNSHAPED_KEY}",
    }

    found = secrets_found(texts, key_values=[FAKE_UNSHAPED_KEY])

    assert found == ["graph/shaped.py: Tavily key", "REVIEW_PACK.md: a real key value"]
    # The message names where, never the key itself.
    assert not any(FAKE_TAVILY_KEY in item or FAKE_UNSHAPED_KEY in item for item in found)


def test_nothing_is_reported_when_there_is_no_key():
    assert secrets_found({"settings.py": "MISTRAL_API_KEY="}, key_values=[]) == []


def test_every_packed_file_that_is_not_python_has_a_summary():
    # A file packed without a summary leaves a blank in the file map.
    for path in INCLUDED_FILES - {"settings.py"}:
        assert module_summary(path, "")
    assert module_summary("graph/a.py", '"""Line one.\n\nMore."""') == "Line one."
    assert module_summary("graph/broken.py", "def (") == ""


def test_the_project_description_is_the_first_plain_paragraph():
    readme = "# Title\n\n> A quote\n\n| a | b |\n\nThe engine researches a topic.\n\nMore."

    assert project_description(readme) == "The engine researches a topic."
    assert project_description("# Only a heading") == ""


def test_code_goes_to_the_same_part_as_its_tests():
    assert area_of("mcp_servers/search_mcp/url_safety.py") == "web"
    assert area_of("tests/unit/test_url_safety.py") == "web"
    assert area_of("graph/self_rag/nodes.py") == "documents"
    assert area_of("tests/unit/test_self_rag_prompts.py") == "documents"
    assert area_of("app/ingestion.py") == "documents"
    assert area_of("graph/run_budget.py") == "graph"
    assert area_of("tests/unit/test_graph_documents.py") == "graph"
    # Anything no area names goes to the last one.
    assert area_of("graph/llm.py") == "core"
    assert area_of("graph/a_new_module.py") == "core"


def test_every_file_is_in_exactly_one_part_and_the_helpers_in_all():
    paths = [
        "graph/llm.py",
        "graph/retrieval.py",
        "mcp_servers/search_mcp/fetch.py",
        "settings.py",
        *SHARED_PATHS,
    ]

    parts = split_into_parts(paths)

    assert list(parts) == ["web", "documents", "core"]
    for part_paths in parts.values():
        assert part_paths[: len(SHARED_PATHS)] == list(SHARED_PATHS)
    not_shared = [path for part in parts.values() for path in part if path not in SHARED_PATHS]
    assert sorted(not_shared) == sorted(set(paths) - set(SHARED_PATHS))


def test_an_area_with_no_files_of_its_own_gets_no_part():
    parts = split_into_parts(["graph/llm.py", "tests/conftest.py"])

    assert list(parts) == ["core"]


def test_a_part_ends_with_a_line_that_names_it_and_counts_its_files():
    part = render_part(
        {"README.md": "Run:\n```\nuv sync\n```\n", "graph/a.py": "x = 1\n"},
        number=2,
        part_count=3,
        title="an area",
    )

    assert part.startswith("# Review files, part 2 of 3: an area\n")
    assert part.rstrip().splitlines()[-1] == "End of part 2 of 3: 2 files."
    # A file holding a code fence gets a longer fence, so it cannot end the block early.
    assert "## `README.md`\n\n````\nRun:\n```\nuv sync\n```\n````" in part
    assert "## `graph/a.py`\n\n```python\nx = 1\n```" in part


def test_parts_are_numbered_in_order_and_named_by_area():
    parts = render_parts(
        {"graph/llm.py": "a = 1\n", "app/ingestion.py": "b = 2\n", "tests/conftest.py": "c\n"}
    )

    assert list(parts) == ["FILES_1_documents.md", "FILES_2_core.md"]
    assert parts["FILES_1_documents.md"].rstrip().endswith("End of part 1 of 2: 2 files.")
    assert "## `tests/conftest.py`" in parts["FILES_2_core.md"]


def test_a_part_over_the_limit_stops_the_build():
    oversized = "x" * (PART_LIMIT_BYTES + 1)

    with pytest.raises(SystemExit, match="FILES_1_core.md is .* KB, over the"):
        render_parts({"graph/llm.py": oversized})
