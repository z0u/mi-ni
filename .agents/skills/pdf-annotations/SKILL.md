---
name: pdf-annotations
description: Read a hand-annotated PDF (a report reviewed in ink on a reMarkable or similar) and turn the marks into a written transcript that a colleague can apply. Use when the human attaches a marked-up PDF, or asks what their annotations say.
---

Sandy reviews reports on a reMarkable: the published page is printed to PDF (narrow, very tall pages, one per section), marked up in ink, and the export is attached to the chat. The ink comes back as vector geometry drawn into the page itself, so there are no annotation objects to extract and no text to search. The only way to read it is to look. This skill makes that cheap and repeatable: a script renders the pages into strips a vision model can read, and the `annotation-transcriber` agent reads them and writes the transcript, so the supervising agent works from text and spends its own context on applying the review.

## Steps

1. **Find the file.** Attachments land under `/root/.claude/uploads/<session-id>/`; `fd -e pdf /root/.claude/uploads` finds it. If the human says they attached a file and nothing is there, they forgot; ask.

2. **Render.** `pymupdf` is not a project dependency, so run the bundled script under `uvx`. In the remote sandbox, PyPI is on the proxy's bypass list and the direct route stalls, so send it through the proxy by clearing the bypass:

   ```bash
   env NO_PROXY=localhost no_proxy=localhost \
     uvx --with pymupdf python .claude/skills/pdf-annotations/render_strips.py review.pdf "$SCRATCHPAD/review"
   ```

   (`uvx` prints a `UV_NATIVE_TLS` deprecation warning; ignore it.) The output directory gets `pNN.txt` (the printed text, for quoting), `pNN_sK.png` (strips at 2x, cropped to the page's content, about 900 pt tall with a 40 pt overlap) and `manifest.md`, a table of every strip with the number and colours of ink strokes on it. Skim the manifest: the strips marked `printed text only` need no reading, and the header says whether the PDF has real annotation objects, in which case extract those with pymupdf instead of rendering. The last review, 14 pages, came to 20 strips and 3 MB.

3. **Transcribe.** Hand the directory to the `annotation-transcriber` agent (Opus). Tell it who the reviewer is and what document was reviewed, in a sentence. It writes `transcript.md` beside the strips and returns a short summary; the file is the deliverable. Expect a few minutes and something under 200k of the agent's tokens for a dozen pages.

4. **Apply.** Work from the transcript, opening a strip yourself only where the transcript flags `[unsure]` or an interpretation seems off. Handwritten review is approximate by nature (the human said as much), so push back on a mark that looks mistaken, answer questions written in the margin, and say in the reply which readings you took where the transcript was unsure. A reviewer's question in the margin usually wants a number or a sentence in the report, and sometimes the answer is "no, and here is why".

## What the ink looks like

The script's heuristics, so you can tell when they mislead: a pen stroke is a filled path of many curve segments in a colour that is neither black nor white; a highlighter stroke is a wide stroked line; anything with a colour that repeats with identical geometry across the document (list bullets, icons) is printed. A margin bar drawn with the tablet's rectangle tool is a single rectangle, so it is reported under `shape colours` next to printed badges and admonition backgrounds. Reviewer colours seen so far: red for edits and notes, purple for document-level marks (a DRAFT badge), green for a section-wide bar, magenta highlighter.

Scale matters for legibility. 2x zoom on a 450 pt wide page gives 900 px strips; handwriting is readable there, and 1x is not. A strip much taller than about 2000 px loses detail when the model views it, which is why the pages are cut.
