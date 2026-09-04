---
status: done
tags: [cli, devx, reports]
closed: 2026-08-23
---
# An easier way to open reports for editing

What works:

1. Launch Marimo in edit mode: `marimo edit --watch --headless docs/.../report.py`
2. Open the editor by clicking on: `URL: http://localhost:2718?access_token=...`
3. Click the "Toggle app view" button in the Marimo UI
4. Edit the source file directly in the IDE instead of Marimo

This then allows humans an AIs to edit the source, and the human can see the changes in the Marimo UI. Preprequisite config (already applied):

```toml
[tool.marimo.runtime]
watcher_on_save = "autorun"
```

The friction is mostly in steps 1 and 3. 1, because it's a long command to have to type. I never use `./go open` or `./go edit` or whatever: they're not the right verbs. They can be retired an new ones added, if that's what makes sense. 3, because you have to find the button with your mouse. Can we add a query param to the URL to automate it? Having to click on the URL in 2 is a bit annoying; it's only necessary because of `--headless` — BUT without `--headless` the URL is opened in the external browser. I want it to open inside VS Code, which it does if I Cmd-click on it.

## Notes

**2026-08-23, tech debt** — Both friction points are gone. `./go open <report>` now runs `marimo edit --watch --headless` through `scripts/edit_notebook.py`, which streams marimo's output and rewrites the one line you want to click:

```
        ➜  App:    http://localhost:2718?access_token=…&view-as=present
        ➜  Editor: http://localhost:2718?access_token=…
```

The query param does exist: marimo's frontend reads `view-as=present`, and in *edit* mode that starts the session in the app view — the same state the toggle button reaches. Verified in a headless browser against a live server rather than from the frontend source: the App link renders the notebook with zero visible editors and no cell-creation chrome, the Editor link renders it with both, and the pencil toggle still switches between them. The watch half was checked the same way, editing the file on disk and waiting for the app view to follow — it does, within a second or two, so `watcher_on_save = "autorun"` is doing its job.

Rewriting marimo's banner rather than printing our own URL is deliberate: marimo keeps choosing its own port and token, so there's nothing to allocate or poll for, and only one link is on screen at the moment you want to click one. The cost is a pipe on marimo's stdout, which is why the child gets `PYTHONUNBUFFERED=1` — a short banner would otherwise sit in a block buffer.

`--browser` opts back out to the old behaviour (marimo opens the tab itself), for anyone not driving this from an editor terminal.

Left alone: the verb. `open` is still `open`, because the note says the current names are wrong without saying what would be right, and a name is a taste call worth making deliberately rather than guessing at. The behaviour behind it is what changed. Renaming later costs one line in `go`, and the retired-verb hints at the foot of that file are the pattern for doing it kindly.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/open-reports-for-editing.md`](https://github.com/z0u/sca2/blob/main/todo/eng/open-reports-for-editing.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.
