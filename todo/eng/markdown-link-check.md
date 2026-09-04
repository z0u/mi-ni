---
status: done
tags: [ci, docs, skills]
opened: 2026-08-22
closed: 2026-08-24
---
# Nothing checks that a relative doc link or `#anchor` still resolves

The skills and `eng/` docs navigate by relative link, and several carry an `#anchor` into a named section (`storage.md#publishing-to-the-web`, `running.md#hotfix-safety-avoid-double-spending`, and two added when the mi-ni reference docs were deduplicated). A renamed heading breaks an anchor silently: the link still opens the file, just at the top, so a session following it lands somewhere plausible and reads the wrong section. A moved file is louder but no more likely to be noticed, since nothing reads these paths but agents.

The check is small — parse each `.md`, resolve every non-`http` link target relative to its own directory, and for a `#fragment` compare against the GitHub-style slugs of that file's headings. Written throwaway during the dedup pass it was about 30 lines and found nothing else broken, so the value is in the next rename rather than the current tree. Two wrinkles it needs: fenced code has to be stripped before matching (`recovery.md`'s table sits in one), and so do inline code spans, since `reports.md` deliberately shows `` `[experiment](./experiment.py)` `` as an example of a link a report author writes. Without that second strip it reports one false positive.

Natural home is a `./go` verb beside `deadcode` and `annotations`, advisory in `lint-check.yml` on the same footing. Worth deciding whether it covers `eng/` and `README.md` too, or only `.agents/skills/`: the wider net catches more, and the docs outside skills link to source files that move more often than headings do.

## Notes

**2026-08-24, tech debt** — Done: `scripts/check_md_links.py`, `./go links`, and a `Doc links` step in `lint-check.yml`. Both wrinkles the body named are real and are covered by tests; blanking a fence rather than deleting it is what keeps a later finding's line number honest.

Two decisions went against the body, and both are cheap to reverse.

**A gate, not advisory.** `dead` and `annotations` are advisory because each reports a backlog longer than one branch should carry; this one arrived at zero across every file we author, so holding it there costs nothing — and advisory output is exactly what nobody reads on the day of the rename this check exists for. The repo-wide assertion also sits in `tests/test_check_md_links.py`, so `./go check` catches it before a push.

**The wider net, minus what we didn't write.** The default is every tracked `.md` except `references/` and `src/subline/`, neither of which this template ships; in the project that wrote the check they carry 21 breaks that were broken on arrival and stay broken however carefully we rename: `references/` is prior-work papers and posts imported as text whose figures never came with them, and `src/subline/` is a vendored README pointing at its own upstream tree. Naming a path explicitly still reaches them (`./go links references/`).

Scope beyond the body: reference definitions (`[n]: ./x.md`) as well as inline links, images, root-absolute `/AGENTS.md`, angle-bracketed and percent-escaped destinations, and explicit `id=`/`name=` HTML anchors. Fragments are checked only on `.md` targets, since `foo.py#L10` is a line reference.

`git ls-files` is the file source, with `--others --exclude-standard` so untracked files come too. That last part started as a bug worth naming: listing only tracked files meant writing a doc, running the check, and being told the tree was clean — and a doc written this session is the likeliest place for a link to be wrong. `.gitignore` still does the excluding, so `.venv/`, `_site/`, `.mini/` and `.claude/worktrees/` need no rule, and the tracked `.claude/skills` symlink can't double-report `.agents/skills`.

Verified end to end rather than by unit tests alone: renaming `## Publishing to the web` and moving `eng/gc.md` produced four findings across three link styles — a skill-to-skill anchor, a sibling `./gc.md`, and a `../../eng/gc.md` from the todo tree — and reverting cleared them.

**2026-08-24, review round** — `links` moved into `./go check --links` and joined its default set, keeping the standalone verb for a targeted run, which is the shape every other check already has.

Also settled where root-absolute links work, since we now prefer them. GitHub rewrites `/foo` to the repo root at the current ref — checked against our own rendered `WELFARE.md`, where `[AGENTS.md](/AGENTS.md)` comes out as `/z0u/mi-ni/blob/main/AGENTS.md`. VS Code resolves the same form against the *workspace* root, which is the repo root on a normal single-folder open; [microsoft/vscode#299488](https://github.com/microsoft/vscode/issues/299488) is an open bug where a `SKILL.md` gets a false "not found" squiggle for exactly this form while preview and ctrl-click follow it correctly, so the diagnostic is the thing that's wrong, not the link.

The exception is `docs/`, and it is the reason the checker now reports a root-absolute link there. `build_site.py` renders `docs/**.md` to HTML for the published site, and `resolve()` bails on any anchored target (`_ANCHORED` matches the leading `/`), so the link is emitted untouched — against a site served from `z0u.github.io/mi-ni/`, it resolves to the domain root and 404s while resolving perfectly on disk and on GitHub. Precisely the quiet failure this item was opened for, and invisible to a check that only asks whether the file exists.

Converted the cross-tree links on that basis, leaving `../` where the relation is the point: the three backlog READMEs naming each other, and `docs/README.md`.

**2026-08-28, follow-up** — The `docs/` exception is gone; root-absolute now works everywhere. `resolve()` reads such a target against the repo root and rebases it onto `docs/`, so it takes the same branches as the relative form: one under `docs/` renders like any other page, one outside it points at the GitHub source. `_ANCHORED` drops the bare `/` and so matches what `mini.reports` already did — scheme, protocol-relative, fragment. `PUBLISHED_ABSOLUTE` and its branch came out of the checker with it.

Prompted by a design doc under `docs/` that arrived with several `/todo/` links written to the new house style, the one place it didn't hold. Converting them to `../../../` would have worked and read against the grain of the preference, so the fix went to the resolver instead. Verified on a real `--externalize` build: they come out as `github.com/<owner>/<repo>/blob/main/...`.

**2026-08-28, follow-up 2** — Heading anchors now work on the site too. `docs/**.md` renders through Python-Markdown's `toc`, so every heading carries an `id`; before this a `#fragment` into a docs page resolved on GitHub and scrolled nowhere on `z0u.github.io/mi-ni/`, and the check couldn't see it because the heading was there in the source all along.

`toc`'s own slugify collapses a run of separators, which would have made a third dialect — `provenance-cost` where GitHub and this check both say `provenance--cost`. So the slug has one definition, `mini.reports.github_slug`, imported by both scripts, with the awkward cases asserted on each side.

Repeats are the one case left: GitHub numbers them `-1`, `toc` numbers them `_1`, and `toc` owns that internally. Rather than teach the fragment check which dialect a target speaks, a duplicate heading under `docs/` is now a finding (`DUPLICATE_HEADING`) — reword it, or give it its own `<span id>`, which the check honours so the remedy it names actually clears it. Outside `docs/` only GitHub renders, so the existing `-1` modelling stands and duplicates are fine.

Consequence for authors: an explicit `<span id>` is no longer needed to make a section linkable, only to make a link survive a rewording. Two went out of `docs/index.md` on that basis.

**2026-09-03, backport** — Ported from sca2 with the code that cites it, so some details above describe that project's tree; the reasoning is what the shared code relies on.
