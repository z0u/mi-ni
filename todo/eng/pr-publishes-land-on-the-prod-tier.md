---
status: open
tags: [publishing, ci]
opened: 2026-07-14
---
# PR publishes land on the prod publish tier

`./go publish` from a PR branch writes `exports/<key>/` on the *production* tier. A new report sits there dark until `main` links it, which is fine (the PR preview even depends on it), but re-publishing an *existing* key from a branch silently swaps the assets under the live site's stale HTML.

If that bites, publish PR exports to a `pr-<n>` git revision of the dataset repo (`upload_folder(revision=...)`, preview `<base>` at `resolve/pr-<n>/`). See [`eng/publishing.md`](/eng/publishing.md).
