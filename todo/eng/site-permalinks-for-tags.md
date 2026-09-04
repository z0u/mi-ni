---
status: open
tags: [ci, publishing]
opened: 2026-08-30
---
# A tag should be a permalink to the site as it stood

Split out of the `gh-pages` branch-pruning work, which asked for it as a rider and got the pruning done first.

We cut a tag when a milestone lands, and it's an indelible release on GitHub. The code at that tag is durable and the report bundles behind it are too (`docs/publish.lock` pins each one to an immutable dataset-repo commit, and the CI build resolves through the pin, so a tag names an exact set of bundles forever). The assembled *site* is the piece that isn't durable: `gh-pages` serves one tree, the tip, and the moment the next report merges there is no URL left for what the site said when the milestone closed.

The shape that fits what's already there: build the site at the tag and deploy it under `v/<tag>/` on `gh-pages`, the way `pr-preview/pr-<n>/` already coexists with production. `./go site` takes `MINI_SITE_URL`, which is how a preview keeps its inter-report links inside its own subtree, so the same switch should give a snapshot self-contained links. Production's deploy step then needs `v/` added to its `clean-exclude` alongside `pr-preview/`, or the next merge to `main` deletes every snapshot.

Two things make this cheaper than it looks. The build is read-only — it assembles HTML around bundles that are already on the publish tier — so a snapshot costs a CI minute and no compute. And history pruning doesn't threaten it: the pruner re-roots the branch but copies the tip's tree verbatim, so files under `v/` survive a prune exactly as the preview trees do. The durability of a snapshot rests on the files at the tip, never on an old commit still resolving.

Open questions worth settling before building it: whether the trigger is `on: push: tags:` or a manual dispatch (tags get cut by hand and rarely, so dispatch may be enough); whether the index at the site root should list the snapshots, or the release notes carry the link; and how much a snapshot's assets actually cost, given the `<base href>` leaves them on the CDN and the pin means two snapshots sharing a report share its bundle rather than copying it.

## Notes

**2026-09-02** — The deploy is now a whole-site rebuild by one writer (`scripts/deploy_site.py`), so there is no `clean-exclude` to add and no prune to survive. A snapshot would instead be one more checkout in that script's loop: build `main` and each open PR as now, plus each tag under `v/<tag>/` with `MINI_SITE_URL` pointed there. Tags are few and the build is read-only, so the cost stays a CI minute each.

**2026-09-03, backport** — Ported from sca2 with the code that cites it, so some details above describe that project's tree; the reasoning is what the shared code relies on.
