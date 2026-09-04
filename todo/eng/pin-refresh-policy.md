---
status: open
tags: [ci]
opened: 2026-08-21
---
# Nothing tells us when a pinned Action goes stale

`astral-sh/setup-uv` is now pinned to a full version (`@v10.0.1`) in both workflows, because upstream stopped publishing floating major tags at v8 — a mutable `@v8` is a supply chain the pin can't see, so naming the release is their recommendation. That trade lands the freshness problem on us: the pin will sit at v10.0.1 until someone edits two files, and nothing in the repo will mention it. The Node 20 pins took about eleven months to get written down (deprecation announced 2025-09-19, item opened 2026-08-19), and that was with a *warning* printed on every single run; an exact pin is quieter than that.

The obvious answer is a `.github/dependabot.yml` with the `github-actions` ecosystem on a monthly interval, which opens a PR per stale pin and understands full-version and SHA pins alike. That is a process change rather than a code one — it means a bot opening PRs on this repo and every one of them spending a CI round — so it wants a human decision rather than a quiet commit. Worth weighing against the alternative of grouping the updates (`groups:` in the config keeps it to one PR per interval), or of doing nothing and re-checking by hand whenever CI is being touched anyway.

Scope is small either way: two pinned Actions across two workflows. `actions/checkout` still publishes floating majors, so today only setup-uv actually needs the reminder — but that is a property of one upstream, not a guarantee, and the same immutable-release argument is spreading.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/pin-refresh-policy.md`](https://github.com/z0u/sca2/blob/main/todo/eng/pin-refresh-policy.md) there) with the code it describes. Here the two workflows are `lint-check.yml` and `site.yml`, and `template-cleanup.yml` still pins `actions/checkout@v4`, which the Node 20 sweep missed in this tree: one line, whenever CI is next touched.
