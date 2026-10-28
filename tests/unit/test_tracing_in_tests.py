"""Scripted tests send no traces to LangSmith (D-41).

CONCEPT: tracing-from-day-one. The free LangSmith plan records 5,000 traces a
month and then stops. Scripted tests call no real model, so a trace of them is
quota spent on nothing. tests/conftest.py turns tracing off for any run that
does not select the live tests; these tests fail if that cap stops working.
"""

import os

import pytest
from langsmith.utils import tracing_is_enabled

from tests.conftest import run_selects_live_tests


def test_scripted_test_run_has_tracing_turned_off():
    # This file is a scripted test, so it runs only when the live tests are
    # not selected: tracing must be off for the whole run.
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert not tracing_is_enabled()


@pytest.mark.parametrize(
    "marker_expression",
    ["live", "live and not slow", "(live)", "slow or live"],
)
def test_expressions_that_select_live_tests_keep_tracing(marker_expression):
    assert run_selects_live_tests(marker_expression)


@pytest.mark.parametrize(
    "marker_expression",
    ["", "not live", "not (live)", "slow", "not live and not slow"],
)
def test_expressions_without_live_tests_turn_tracing_off(marker_expression):
    assert not run_selects_live_tests(marker_expression)
