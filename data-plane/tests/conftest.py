"""pytest configuration for the data-plane test suite."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: requires Cerebras + Deepgram API keys — skipped in offline/CI runs",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.live tests when --live flag is not passed."""
    if config.getoption("--live", default=False):
        return
    skip_live = pytest.mark.skip(reason="live LLM test — run with --live to enable")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that require live Cerebras/Deepgram API calls",
    )
