#!/usr/bin/env python
"""Report relative Markdown links and `#anchor` fragments that no longer resolve.

The skills and `eng/` docs navigate by relative link, and several carry a fragment into a named section. Nothing reads these paths but agents, so both failure modes are quiet: a moved file gives a dead link nobody clicks, and a renamed heading is worse — the link still opens the file, just at the top, so a session following it lands somewhere plausible and reads the wrong section. This makes both loud at the point the rename happens.

Only *relative* targets are checked. An `http(s)://` or `mailto:` link needs the network to verify and goes stale for reasons outside this repo, which is a different job with a different failure rate.

Two strips before any matching, both load-bearing. Fenced code blocks hold illustrative fragments that contain bracket-paren pairs — a backlog note's repro has `(x: Sequence[T])`, which reads as a link target otherwise. Inline code spans hold links quoted *as examples*: `.agents/skills/mi-ni/references/reports.md` deliberately shows ``[experiment](./experiment.py)`` as the shape a report author writes, and without the strip it is reported against whichever directory the doc happens to sit in.

Anchors are matched against GitHub's slugs, since that is where these docs are read — `mini.reports.github_slug`, the same function `build_site.py` hands to Python-Markdown so a published page carries the ids GitHub would give it. So "Provenance & cost" is `#provenance--cost`: the removed `&` leaves the two spaces that become two hyphens. Explicit `id=`/`name=` attributes on inline HTML count as anchors too, which is how a hand-written target survives a heading rewrite.

Repeats are where the two renderers part company, GitHub numbering them `-1` and `toc` `_1`. Under `docs/`, which is rendered both ways, a duplicate heading is therefore reported rather than resolved to one dialect or the other; elsewhere GitHub's numbering is the only one in play and stands.

The file set comes from `git ls-files`, including untracked-but-not-ignored files — a doc written this session is the likeliest place for a link to be wrong, so listing only what git already knows about would hand it a clean bill of health. `.gitignore` does the excluding, so `.venv/`, `_site/`, `.mini/` and `.claude/worktrees/` drop out without a rule of their own, and `.claude/skills` is a tracked *symlink* to `.agents/skills`, so git lists the real files once and the walk can't double-report through it.
"""

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from mini.reports import github_slug

ROOT = Path(__file__).parent.parent.resolve()

# Skipped unless named on the command line, because their links describe a layout that isn't
# ours: `references/` holds prior-work papers and posts imported as text, whose figures were
# never copied in, and `src/subline/` is a vendored library whose README points at its own
# upstream tree. Both were broken on arrival and stay broken however carefully we rename —
# 21 findings that would drown the one this check exists for. `./go links references/` still
# reaches them.
UNOWNED = ("references/", "src/subline/")

MISSING_FILE = "no such file"

# Two headings that slug alike are fine on GitHub, which numbers the repeats `-1`, `-2`.
# `docs/` is the one tree also rendered to the published site, and Python-Markdown's `toc`
# numbers them `_1`, `_2` — so the second heading's anchor differs between the two places a
# reader might follow it, and only one of them can be what a checked `#fragment` matched.
# Rather than teach the fragment check which dialect a target speaks, we keep `docs/` free of
# the ambiguity. A heading carrying its own `id=` is already unambiguous and doesn't count.
DUPLICATE_HEADING = (
    "duplicate heading under docs/, which GitHub and the site number differently"
    ' — reword it, or give it its own <span id="..."></span>'
)

# A fence is three-or-more backticks or tildes; the closer must be at least as long and of
# the same kind, so a ````-fenced block can quote a ``` one (the docs do this).
FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")

# Inline code: a run of N backticks closed by the next run of exactly N. Matching longest-first
# keeps ``a ` b`` from being read as two single-backtick spans.
CODE_SPAN = re.compile(r"(?P<ticks>`+)(?P<body>.+?)(?P=ticks)", re.DOTALL)

# `[text](target)` and `![alt](src)`. An unbracketed destination stops at whitespace, so a
# `(path "Title")` link keeps only the path — but `<...>` has to be matched first and whole,
# since angle brackets are exactly what Markdown offers for a destination containing spaces.
INLINE_LINK = re.compile(r"!?\[(?P<text>[^\]]*)\]\(\s*(?P<dest><[^>]*>|[^\s)]*)[^)]*\)")

# A link *reference definition* at the head of a line: `[label]: target "Title"`. Excludes
# footnotes (`[^id]:`), which are prose bodies rather than targets.
REF_DEF = re.compile(r"^ {0,3}\[(?!\^)(?P<label>[^\]]+)\]:[ \t]*(?P<dest><[^>]*>|\S+)", re.MULTILINE)

# ATX headings only. Setext (`===` underlines) appears nowhere in this tree, and guessing at
# one would mean re-implementing the paragraph rules that decide whether it is a heading.
HEADING = re.compile(r"^(?P<indent> {0,3})(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*#*\s*$", re.MULTILINE)

# `id="x"` / `name="x"` on any inline HTML — an anchor a heading rename can't take away.
HTML_ANCHOR = re.compile(r"""<[^>]*\b(?:id|name)\s*=\s*["'](?P<anchor>[^"']+)["'][^>]*>""")

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Anything with a scheme (`https:`, `mailto:`) or protocol-relative (`//host`) leaves the repo.
EXTERNAL = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")


def display(path: Path) -> str:
    """Repo-relative where that's meaningful — the usual case, and what an editor links on.

    As given otherwise: a `../` link can resolve to a real file outside the checkout, and a path passed from outside it still has to print rather than raise.
    """
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


@dataclass(frozen=True, order=True)
class Finding:
    """One relative link whose target, or whose fragment, doesn't resolve."""

    path: Path
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{display(self.path)}:{self.line}: {self.target} — {self.reason}"


def strip_code(text: str) -> str:
    """Blank out fenced blocks and inline spans, keeping every newline so line numbers survive.

    Replacement rather than deletion is the whole point: a finding reports the line it was found on, and deleting a thirty-line fence would shift every line after it.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        if (m := FENCE.match(line)) is not None:
            if fence is None:
                fence = m["fence"]
                out.append("")
                continue
            # A closer matches the opener's kind and is at least as long; anything else is content.
            if m["fence"][0] == fence[0] and len(m["fence"]) >= len(fence) and not m["info"].strip():
                fence = None
                out.append("")
                continue
        out.append("" if fence is not None else line)
    # Spans last, so a backtick inside a fenced block can't pair with one outside it.
    return CODE_SPAN.sub(lambda m: "\n" * m.group(0).count("\n"), "\n".join(out))


def anchors_in(text: str) -> set[str]:
    """Every fragment *text* offers: one slug per heading, plus explicit HTML `id`/`name`.

    Repeats take GitHub's `-1`, `-2` suffixes, in document order — and the bare slug stays valid for the first of them, so both forms are offered.
    """
    body = strip_code(HTML_COMMENT.sub("", text))
    found: set[str] = set()
    seen: dict[str, int] = {}
    for m in HEADING.finditer(body):
        if not (slug := github_slug(m["text"])):
            continue  # a heading of pure punctuation gets no anchor from GitHub either
        count = seen.get(slug, 0)
        found.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    found.update(m["anchor"] for m in HTML_ANCHOR.finditer(text))
    return found


def _targets(body: str) -> Iterator[tuple[int, str]]:
    """Every link target in a code-stripped document, with the 1-based line it sits on."""
    for pattern in (INLINE_LINK, REF_DEF):
        for m in pattern.finditer(body):
            yield body.count("\n", 0, m.start()) + 1, m["dest"]


def _resolve(target: str, source: Path) -> Path:
    """Where *target* points, from a link written in *source*.

    A leading `/` means the repo root rather than the filesystem root — that is how it reads on GitHub, and how the two such links in this tree are meant.
    """
    return ROOT / target.lstrip("/") if target.startswith("/") else (source.parent / target)


def _duplicate_headings(path: Path, body: str) -> list[Finding]:
    """Headings in a code-stripped *body* that slug the same as an earlier one, minus those carrying an explicit `id=`.

    Reported for `docs/` only — see :data:`DUPLICATE_HEADING`. The first of a run stands; each one after it is a finding, so the fix is local to the heading that arrived second.
    """
    found: list[Finding] = []
    seen: set[str] = set()
    for m in HEADING.finditer(body):
        if not (slug := github_slug(m["text"])):
            continue
        if slug in seen and not HTML_ANCHOR.search(m["text"]):
            found.append(Finding(path, body[: m.start()].count("\n") + 1, m["text"].strip(), DUPLICATE_HEADING))
        seen.add(slug)
    return found


def findings_in(path: Path, cache: dict[Path, set[str]]) -> list[Finding]:
    """Every relative link in *path* whose file is missing or whose fragment names no heading."""
    text = path.read_text("utf-8", errors="ignore")
    body = strip_code(HTML_COMMENT.sub("", text))
    found: list[Finding] = []

    if path.resolve().is_relative_to(ROOT / "docs"):
        found += _duplicate_headings(path, body)

    for line, raw in _targets(body):
        target = unquote(raw.strip().removeprefix("<").removesuffix(">"))
        if not target or EXTERNAL.match(target):
            continue

        path_part, _, fragment = target.partition("#")
        dest = path.resolve() if not path_part else _resolve(path_part, path).resolve()

        if not dest.exists():
            found.append(Finding(path, line, target, MISSING_FILE))
            continue
        if not fragment or dest.suffix != ".md":
            continue  # a fragment on a non-Markdown target is a line ref or a viewer's business

        if dest not in cache:
            cache[dest] = anchors_in(dest.read_text("utf-8", errors="ignore"))
        if unquote(fragment) not in cache[dest]:
            where = "in this file" if dest == path.resolve() else f"in {display(dest)}"
            found.append(Finding(path, line, target, f"no heading {where} slugs to #{unquote(fragment)}"))

    return sorted(found)


def repo_markdown(paths: Iterable[Path]) -> list[Path]:
    """The `.md` files under *paths* that belong to this repo — git decides what that means.

    `--others --exclude-standard` puts *untracked* files in alongside tracked ones, which is what makes the check useful before a commit: a doc written this session is the likeliest place for a link to be wrong, and listing only what git already knows about would answer it with a clean bill of health. `.gitignore` still does the excluding, so `.venv/`, `_site/`, `.mini/` and `.claude/worktrees/` drop out without a rule of their own.

    :data:`UNOWNED` is filtered out only for the default (whole-repo) scope: naming a path is a deliberate act, and answering it with silence would be worse than the noise.
    """
    args = [str(p) for p in paths]
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *(args or ["."]),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = (n for n in listed.split("\0") if n.endswith(".md"))
    if not args:
        names = (n for n in names if not n.startswith(UNOWNED))
    return sorted(ROOT / name for name in names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="files or directories to check (default: the whole repo)")
    args = ap.parse_args()

    if missing := [p for p in args.paths if not p.exists()]:
        sys.exit(f"no such path: {', '.join(p.as_posix() for p in missing)}")
    # git resolves a pathspec against the repo, so one from outside it is an error there rather
    # than an empty result — caught here so it reads as a usage mistake, not a crash.
    if outside := [p for p in args.paths if not p.resolve().is_relative_to(ROOT)]:
        sys.exit(f"outside the repo: {', '.join(p.as_posix() for p in outside)}")

    targets = repo_markdown(args.paths)
    cache: dict[Path, set[str]] = {}
    found = sorted(f for path in targets for f in findings_in(path, cache))

    if not found:
        print(f"✅ Every relative link and #anchor resolves, and docs/ headings are unambiguous ({len(targets)} files)")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    counts = {
        "missing file": sum(1 for f in found if f.reason == MISSING_FILE),
        "unresolved #anchor": sum(1 for f in found if f.reason.startswith("no heading")),
        "duplicate heading under docs/": sum(1 for f in found if f.reason == DUPLICATE_HEADING),
    }
    print(  # the tally is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} problem(s) across {len(targets)} files — "
        + ", ".join(f"{n} {label}" for label, n in counts.items() if n),
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
