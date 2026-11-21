"""The test inventory's pure parts: the testing folder's phase files.

CONCEPT: the testing folder (docs repo, `testing/`). scripts/make_test_inventory.py
reads the tests as they are written and git history, which these tests leave
alone; they check what decides each phase file: which commit added a test,
which phase a commit belongs to, how a test is described, and that the part
written by hand is never overwritten.
"""

from scripts.make_review_pack import InventoryEntry
from scripts.make_test_inventory import (
    GENERATED_MARKER,
    CommitInfo,
    InventoriedTest,
    OriginOfTest,
    descriptions_of_tests,
    first_added_commits,
    last_phase_tag_before,
    merge_into_file,
    origin_of,
    render_generated_part,
    shortened,
)

GIT_LOG = """COMMIT aaa111 [P1 setup] Added the first tests
diff --git a/tests/unit/test_alpha.py b/tests/unit/test_alpha.py
+++ b/tests/unit/test_alpha.py
+def test_one():
+    assert True
+def helper_not_a_test():
+    pass
COMMIT bbb222 [P2 mcp-client] Added a second test and touched the first
diff --git a/tests/unit/test_alpha.py b/tests/unit/test_alpha.py
+++ b/tests/unit/test_alpha.py
-def test_one():
+def test_one():
+async def test_two():
+    assert True
COMMIT ccc333 [review] Moved test_two to a file of its own
+++ b/tests/unit/test_beta.py
+async def test_two():
"""


def test_the_first_commit_that_adds_a_test_is_its_origin():
    by_file_and_name, by_name = first_added_commits(GIT_LOG)

    # A later commit that rewrites the line does not move where the test began.
    assert by_file_and_name[("tests/unit/test_alpha.py", "test_one")].commit == "aaa111"
    assert by_file_and_name[("tests/unit/test_alpha.py", "test_two")].commit == "bbb222"
    # In a new file, the test counts as added there; by name alone, where it began.
    assert by_file_and_name[("tests/unit/test_beta.py", "test_two")].commit == "ccc333"
    assert by_name["test_two"].commit == "bbb222"
    assert "helper_not_a_test" not in by_name
    assert by_name["test_one"].subject == "[P1 setup] Added the first tests"


# Commits in order: positions 0 to 9; phase-01 tagged at 3, phase-02 at 6.
TAG_POSITIONS = {1: 3, 2: 6}


def test_the_last_phase_tag_before_a_commit():
    assert last_phase_tag_before(2, TAG_POSITIONS) == 0
    assert last_phase_tag_before(3, TAG_POSITIONS) == 1  # the tagged commit itself
    assert last_phase_tag_before(5, TAG_POSITIONS) == 1
    assert last_phase_tag_before(9, TAG_POSITIONS) == 2


def test_a_commit_prefix_names_its_phase():
    origin = origin_of(
        CommitInfo("abc", "[P3 shared-throttle] Added the limiter"), 9, TAG_POSITIONS
    )

    # The prefix wins over the tags: work on Phase 3 is Phase 3's.
    assert (origin.phase, origin.at_review) == (3, False)


def test_a_review_commit_belongs_to_the_close_of_the_last_tagged_phase():
    origin = origin_of(CommitInfo("abc", "[review] Tested a fix"), 7, TAG_POSITIONS)

    assert (origin.phase, origin.at_review) == (2, True)


def test_any_other_commit_belongs_to_the_phase_after_the_last_tag():
    after_phase_one = origin_of(CommitInfo("abc", "Updated the file"), 4, TAG_POSITIONS)
    before_any_tag = origin_of(CommitInfo("abc", "Initial commit"), 0, TAG_POSITIONS)
    phase_zero_setup = origin_of(CommitInfo("abc", "[P0 setup] Added CI"), 0, TAG_POSITIONS)

    assert after_phase_one.phase == 2
    assert before_any_tag.phase == 1
    # Phase 0 was the set-up before Phase 1; its tests are listed with Phase 1.
    assert phase_zero_setup.phase == 1


TEST_SOURCE = '''
def test_with_a_docstring():
    """The widget turns blue.

    A second paragraph, left out.
    """


def test_with_a_comment_block():
    # CONCEPT: some-concept, a review note (F-9). The first line of the check
    # carries on here.
    value = 1
    # A later comment describes one step only.
    assert value


def test_with_only_a_concept_line():
    # CONCEPT: some-concept, D-61.
    assert True


async def test_with_nothing():
    assert True
'''


def test_a_test_is_described_by_how_it_is_written():
    descriptions = descriptions_of_tests(TEST_SOURCE)

    assert descriptions == {
        "test_with_a_docstring": "The widget turns blue.",
        # The whole comment block, joined, without the CONCEPT sentence.
        "test_with_a_comment_block": "The first line of the check carries on here.",
        # With only a CONCEPT line, or nothing, the name says it.
        "test_with_only_a_concept_line": "with only a concept line",
        "test_with_nothing": "with nothing",
    }


def test_a_long_description_is_cut_at_a_word():
    assert shortened("one two three", max_chars=20) == "one two three"
    assert shortened("one  two\nthree", max_chars=20) == "one two three"
    assert shortened("alpha beta gamma delta", max_chars=12) == "alpha beta ..."


def entry(name: str, live: bool = False) -> InventoryEntry:
    return InventoryEntry(name=name, description=f"checks {name}", live=live, parametrized=False)


def test_the_generated_part_groups_tests_by_when_and_by_file():
    built = OriginOfTest(phase=4, at_review=False, commit="aaa", subject="[P4 self-rag] x")
    reviewed = OriginOfTest(phase=4, at_review=True, commit="bbb", subject="[review] y")
    tests = [
        InventoriedTest("tests/unit/test_b.py", entry("test_built_in_b"), built),
        InventoriedTest("tests/unit/test_a.py", entry("test_built_in_a"), built),
        InventoriedTest("tests/live/test_c.py", entry("test_live_one", live=True), built),
        InventoriedTest("tests/unit/test_a.py", entry("test_from_the_review"), reviewed),
        InventoriedTest("tests/unit/test_a.py", entry("test_not_committed"), None),
    ]

    generated = render_generated_part(4, tests, head_commit="abc1234")

    assert generated.startswith(GENERATED_MARKER)
    assert "5 test functions first added in this phase: 4 scripted, 1 live" in generated
    assert "at commit `abc1234`" in generated
    built_part, review_part = generated.split("### Added at the Phase 4 close")
    # Files in path order inside each group.
    assert built_part.index("tests/live/test_c.py") < built_part.index("tests/unit/test_a.py")
    assert built_part.index("tests/unit/test_a.py") < built_part.index("tests/unit/test_b.py")
    assert "- `test_live_one` (live): checks test_live_one" in built_part
    assert "test_from_the_review" not in built_part
    assert "- `test_from_the_review`: checks test_from_the_review" in review_part
    assert "### Not yet committed" in review_part
    assert "- `tests/unit/test_a.py`: `test_not_committed`" in review_part


def test_with_no_review_tests_there_is_no_review_heading():
    built = OriginOfTest(phase=2, at_review=False, commit="aaa", subject="[P2 mcp-client] x")

    generated = render_generated_part(
        2, [InventoriedTest("tests/unit/test_a.py", entry("test_one"), built)], "abc1234"
    )

    assert "### Added while building Phase 2" in generated
    assert "### Added at the Phase 2 close" not in generated


def test_the_hand_written_part_is_kept_and_the_generated_part_replaced():
    hand_written = "# Phase 2: testing\n\nWhat these tests prove, written by hand.\n\n"
    old_file = hand_written + GENERATED_MARKER + "\n\nAn old list.\n"
    new_generated = GENERATED_MARKER + "\n\nThe new list.\n"

    merged = merge_into_file(old_file, 2, new_generated)

    assert merged == hand_written + GENERATED_MARKER + "\n\nThe new list.\n"


def test_a_new_file_gets_a_placeholder_summary():
    merged = merge_into_file(None, 5, GENERATED_MARKER + "\n\nThe list.\n")

    assert merged.startswith("# Phase 5: testing\n")
    assert "written by hand at the phase close" in merged
    assert merged.endswith(GENERATED_MARKER + "\n\nThe list.\n")


def test_a_file_without_the_marker_keeps_all_its_text():
    merged = merge_into_file("# Phase 3: testing\n\nNotes.", 3, GENERATED_MARKER + "\n\nList.\n")

    assert merged == "# Phase 3: testing\n\nNotes.\n\n" + GENERATED_MARKER + "\n\nList.\n"
