---
status: done
tags: [ci]
opened: 2026-08-19
closed: 2026-08-21
---
# Two pinned Actions still target Node 20

Every workflow run now ends with a warning: "Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: `actions/checkout@v4`, `astral-sh/setup-uv@v6`." The runner is already substituting Node 24, so nothing is broken today — what expires is the substitution, and the warning is what's left of the notice period ([GitHub changelog, 2025-09-19](https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/)).

Both pins appear three times each, once per workflow (`lint-check.yml`, `pr-preview.yml`, `publish-docs.yml`), so this is six lines and a green run to confirm. `rossjrw/pr-preview-action@v1` and `JamesIves/github-pages-deploy-action@v4` are not named in the warning.

Noticed while reading a CI log for something else, from a session whose GitHub scope was this repo only — so the major versions to move to weren't checked against the upstream release pages. Do that first rather than assuming the next integer, and prefer landing it in a PR that already touches CI, since the confirmation is one full run either way.

## Notes

**2026-08-21, tech debt** — Checked both upstreams by reading `action.yml` at each tag rather than the release prose. `actions/checkout` moved to `node24` at v5 and still publishes floating majors, so it goes to `@v7` (latest; v6 moved persisted credentials into a file under `$RUNNER_TEMP` and v7 blocks fork checkout under `pull_request_target`/`workflow_run` — we use neither trigger, and both deploy actions authenticate with their own token input rather than the checkout's credentials). `astral-sh/setup-uv` moved to `node24` at v7, and at v8 stopped publishing major/minor tags altogether, so `@v6` becomes the full version `@v10.0.1`. We pass it no inputs, so the v9/v10 cache-default changes don't reach us. The exact pin creates a manual-bump obligation, which is now [its own item](./pin-refresh-policy.md).

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/actions-still-on-node-20.md`](https://github.com/z0u/sca2/blob/main/todo/eng/actions-still-on-node-20.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; here the pins live in `lint-check.yml` and `site.yml`.
