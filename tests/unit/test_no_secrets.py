"""TC-NFR-09: no secret in any file git tracks.

CONCEPT: tracing-from-day-one. The code repository is public. This test reads
every file git tracks, which is everything that can reach GitHub, and fails
if it finds:
- a string shaped like a Tavily or LangSmith key, or an API key assignment
  with a real-looking value (works in CI, where the real keys are absent);
- the actual value of any key in this environment (works locally, and
  catches keys with no recognisable shape, such as Mistral's);
- .env itself in the tracked list;
- a committed notebook that still has outputs (nbstripout not installed on
  the machine that committed it).
It scans tracked files, not the folder: .env is supposed to be in the folder.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KEY_SHAPES = {
    "Tavily key": re.compile(r"tvly-[A-Za-z0-9_-]{10,}"),
    "LangSmith key": re.compile(r"lsv2_(?:pt|sk)_[A-Za-z0-9_]{10,}"),
    # [ \t]* rather than \s*: \s also matches a line break, so an empty
    # "MISTRAL_API_KEY=" in .env.example ran on into the next line's name
    # (LANGSMITH_TRACING) and looked like a 17-character value.
    "API key assignment": re.compile(
        r"(?:MISTRAL|TAVILY|LANGSMITH)_API_KEY[ \t]*[=:][ \t]*[\"']?[A-Za-z0-9_-]{16,}"
    ),
}
# Real keys are long; placeholders such as "ci-placeholder" are kept shorter
# than this so they can never match by accident.
MIN_REAL_KEY_LENGTH = 20


def tracked_files() -> list[Path]:
    git_listing = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return [PROJECT_ROOT / relative_path for relative_path in git_listing.stdout.splitlines()]


def tracked_text(file_path: Path) -> str:
    return file_path.read_bytes().decode("utf-8", errors="ignore")


def real_key_values() -> list[str]:
    key_values = [
        settings.mistral_api_key.get_secret_value(),
        settings.tavily_api_key.get_secret_value(),
        settings.langsmith_api_key.get_secret_value(),
    ]
    return [key_value for key_value in key_values if len(key_value) >= MIN_REAL_KEY_LENGTH]


def test_key_shapes_match_real_looking_keys_but_not_empty_templates():
    empty_template = "MISTRAL_API_KEY=\nLANGSMITH_API_KEY=\nLANGSMITH_TRACING=true\n"
    assert not any(pattern.search(empty_template) for pattern in KEY_SHAPES.values())

    # Fake keys are built at run time, so this file never contains a
    # key-shaped string itself and does not trip its own scan.
    fake_assignment = "MISTRAL_API_KEY=" + "a" * 20
    assert KEY_SHAPES["API key assignment"].search(fake_assignment)
    assert KEY_SHAPES["Tavily key"].search("tvly-" + "a" * 20)
    assert KEY_SHAPES["LangSmith key"].search("lsv2_pt_" + "a" * 20)


def test_tc_nfr_09_env_file_is_not_tracked():
    tracked_names = {
        file_path.relative_to(PROJECT_ROOT).as_posix() for file_path in tracked_files()
    }
    assert ".env" not in tracked_names


def test_tc_nfr_09_no_key_shaped_string_in_tracked_files():
    findings_of_keys = [
        f"{file_path.relative_to(PROJECT_ROOT)}: {shape_name}"
        for file_path in tracked_files()
        for shape_name, key_pattern in KEY_SHAPES.items()
        if file_path.is_file() and key_pattern.search(tracked_text(file_path))
    ]
    assert findings_of_keys == []


def test_tc_nfr_09_no_real_key_value_in_tracked_files():
    key_values = real_key_values()
    if not key_values:
        pytest.skip("no real keys in this environment (CI uses short placeholders)")
    files_with_a_key = [
        str(file_path.relative_to(PROJECT_ROOT))
        for file_path in tracked_files()
        if file_path.is_file()
        and any(key_value in tracked_text(file_path) for key_value in key_values)
    ]
    assert files_with_a_key == []


def test_committed_notebooks_have_no_outputs():
    notebooks_with_outputs = []
    for file_path in tracked_files():
        if file_path.suffix != ".ipynb" or not file_path.is_file():
            continue
        # Check the committed version, not the working copy: on disk the
        # notebook keeps its outputs; nbstripout removes them at commit time.
        relative_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        committed_notebook = subprocess.run(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if committed_notebook.returncode != 0:
            continue  # staged but not yet committed
        notebook = json.loads(committed_notebook.stdout)
        if any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"):
            notebooks_with_outputs.append(relative_path)
    assert notebooks_with_outputs == []
