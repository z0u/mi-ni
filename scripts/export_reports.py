#!/usr/bin/env python
"""Export report notebooks to self-contained bundles, optionally syncing to the bucket.

Each report (a ``docs/**/*.py`` declaring ``marimo.App(``) exports to its own bundle at
``.mini/exports/<key>/`` — ``index.html`` plus the named-keyed ``_assets/`` its setup
cell's :func:`~mini.reports.report_bundle` publisher wrote. With ``--publish`` each
bundle is then mirrored to the configured HF bucket at ``exports/<key>/``: the
authenticated half of publishing (it needs the data the report reads + a write token).
``scripts/build_site.py`` assembles the site from these bundles — the synced ones in CI
(read-only), the local ones offline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `import clean_docs` (sibling) works

from clean_docs import clean_html  # noqa: E402
from mini.reports import export_dir, export_key, is_report_notebook, report_notebooks  # noqa: E402

ROOT = Path(__file__).parent.parent.resolve()
DOCS = ROOT / "docs"


def notebooks_to_export(paths: list[str]) -> list[Path]:
    """The report notebooks to export — the given ones, or every report under ``docs/``.

    Source-only example notebooks (``# mini:source-only``, e.g. ``docs/gpt.py``) are
    skipped even when named explicitly: the site links to their GitHub source rather than
    running them, so exporting one (which re-runs its inline compute) is never intended.
    """
    if not paths:
        return report_notebooks(DOCS)
    keep = []
    for p in paths:
        path = Path(p).resolve()
        if is_report_notebook(path):
            keep.append(path)
        else:
            print(f"  skip {path.name}: source-only example, not a rendered report — open it with `./go open`")
    return keep


def export_one(nb: Path) -> Path:
    """Export *nb* to ``.mini/exports/<key>/index.html`` (assets land beside it). Returns the dir."""
    out = export_dir(nb) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  export {nb.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
    subprocess.run(["marimo", "export", "html", "-f", str(nb), "-o", str(out)], check=True, cwd=ROOT)
    clean_html(out)  # scrub terminal control seqs + redact modal URLs from the published HTML
    return out.parent


def publish_one(nb: Path, store) -> None:
    """Export *nb* and mirror its bundle to the bucket at ``exports/<key>/``."""
    bundle = export_one(nb)
    key = export_key(nb)
    print(f"  sync   {bundle.relative_to(ROOT)} -> exports/{key}/")
    store.sync_export(bundle, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true", help="mirror each bundle to the HF bucket after exporting")
    ap.add_argument("notebooks", nargs="*", help="report notebooks (default: all under docs/)")
    args = ap.parse_args()

    nbs = notebooks_to_export(args.notebooks)
    if not nbs:
        sys.exit("No report notebooks found under docs/.")

    if not args.publish:
        for nb in nbs:
            export_one(nb)
        print("\nExported locally to .mini/exports/. Preview with `./go build` (localize) or `./go serve`.")
        return

    from mini.hf_store import HFStore
    from mini.store import store_for

    store = store_for(ROOT / ".mini" / "store")
    if not isinstance(store, HFStore):
        sys.exit("No HF bucket configured — set [tool.mini] store-bucket and run `./go auth`, then retry --publish.")
    for nb in nbs:
        publish_one(nb, store)
    target = store.publish_repo or store.bucket  # exports route to the repo when a publish tier is set (#38)
    print(
        f"\nPublished {len(nbs)} report(s) to {target}. "
        'Trigger the Pages build to update the site (push to main, or run the "Deploy Docs" workflow).'
    )


if __name__ == "__main__":
    main()
