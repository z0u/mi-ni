"""Tests for the live-editing launcher — rewriting marimo's URL banner into an app-view link."""

import pytest

from tests.conftest import load_script

edit_notebook = load_script("edit_notebook")

# Exactly what `marimo edit --headless --watch` prints on stdout (marimo 0.23): eight
# spaces, an arrow, two spaces. The parse is an integration assumption, so the fixture
# is a transcript rather than a paraphrase.
BANNER = "        ➜  URL: http://localhost:2718?access_token=Xk3p9\n"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:2718?access_token=Xk3p9", "http://localhost:2718?access_token=Xk3p9&view-as=present"),
        ("http://localhost:2718", "http://localhost:2718?view-as=present"),
    ],
)
def test_app_view_keeps_the_access_token(url: str, expected: str):
    """The token is the only thing standing between the link and a 403, so it has to survive."""
    assert edit_notebook.app_view(url) == expected


def test_rewrite_offers_the_app_view_first():
    assert edit_notebook.rewrite(BANNER) == (
        "        ➜  App:    http://localhost:2718?access_token=Xk3p9&view-as=present\n"
        "        ➜  Editor: http://localhost:2718?access_token=Xk3p9"
    )


def test_rewrite_reads_a_coloured_banner():
    """marimo drops colour when its stdout is a pipe, which is how we run it — but the parse shouldn't depend on that."""
    coloured = "        ➜  URL: \x1b[1;32mhttp://localhost:2718?access_token=Xk3p9\x1b[0m\n"
    assert edit_notebook.rewrite(coloured) == edit_notebook.rewrite(BANNER)


@pytest.mark.parametrize(
    "line",
    [
        "        Edit report.py in your browser 📝\n",
        "[W 260823 16:13:31 start:271] Enabling watch mode may interfere with auto-save.\n",
        "  Fetched URL: not-a-url\n",
    ],
)
def test_rewrite_passes_everything_else_through(line: str):
    """Anything but the banner is marimo's own output, and is reprinted verbatim."""
    assert edit_notebook.rewrite(line) is None


def test_command_watches_headlessly_and_forwards_the_rest():
    assert edit_notebook.marimo_command(["report.py", "--port", "2718"])[2:] == [
        "marimo",
        "edit",
        "--watch",
        "--headless",
        "report.py",
        "--port",
        "2718",
    ]


@pytest.mark.parametrize(("returncode", "status"), [(0, 0), (-2, 130)])
def test_signalled_marimo_exits_the_way_a_shell_reads_it(returncode: int, status: int):
    """Ctrl-C is SIGINT (2), so the caller should see 130 — not the low byte of -2."""
    assert edit_notebook.exit_status(returncode) == status


def test_browser_flag_is_ours_and_stops_at_us():
    """`--browser` drops `--headless` so marimo opens the tab, and must not reach marimo, which has no such flag."""
    cmd = edit_notebook.marimo_command(["--browser", "report.py"])
    assert "--headless" not in cmd
    assert "--browser" not in cmd
    assert cmd[2:] == ["marimo", "edit", "--watch", "report.py"]
