---
status: open
tags: [publishing, storage]
opened: 2026-07-14
---
# Publish-tier exports go stale on rename

`export_key` derives from the docs-relative path, so moving a notebook orphans its synced bundle: the build looks for the new key, skips with a warning, and the site 404s while `index.md` still links the page. Prevention: teach `./go publish` (or the build) to list remote export keys and warn on ones with no matching notebook, and/or a `./go publish --move old new` verb. Consider folding orphan cleanup into `mini gc --store`.

Same shape one level down: deleting a figure from a notebook leaves its `_assets/<name>-{light,dark}.png` behind, since re-export writes into the existing bundle dir without pruning. Harmless locally (the HTML stops referencing them) but it ships dead bytes on publish — a prune of assets not referenced by the fresh `index.html` would cover both.
