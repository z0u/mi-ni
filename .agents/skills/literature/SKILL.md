---
name: literature
description: How to search and read literature (arXiv papers and the like).
---

## Search

Start broad, then refine.

1. Search using: site:semanticscholar.org [relevant terms]
2. arXiv: "site:arxiv.org [terms]" for recent preprints
3. LessWrong: "site:lesswrong.com [terms]" for AI safety discussions
4. Technical blogs: "site:distill.pub [terms]" or other prominent domains

## Full text

For arXiv, try token-efficient routes first. From most- to least-efficient:

1. Markdown: Use `markxiv.org`. `https://markxiv.org/abs/1706.03762` — the `abs/` path serves the full text as Markdown.
2. HTML: `https://arxiv.org/html/1706.03762`
3. TeX: `https://arxiv.org/src/1706.03762` — decompress in `/tmp` or `scratch-lit/`
4. PDF: `https://arxiv.org/pdf/1706.03762`
