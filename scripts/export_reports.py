#!/usr/bin/env python
"""Export report notebooks to self-contained bundles, optionally syncing to the bucket.

Each report (a ``docs/**/*.py`` declaring ``marimo.App(``) exports to its own bundle at ``.mini/exports/<key>/`` — ``index.html`` plus the named-keyed ``_assets/`` its setup cell's :func:`~mini.reports.report_bundle` publisher wrote. With ``--publish`` each bundle is then mirrored to the configured HF bucket at ``exports/<key>/``: the authenticated half of publishing (it needs the data the report reads + a write token). ``scripts/build_site.py`` assembles the site from these bundles — the synced ones in CI (read-only), the local ones offline.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `import clean_docs` (sibling) works

from clean_docs import clean_html, default_hidden_code  # noqa: E402
from mini.reports import (  # noqa: E402
    EXPORTING_ENV,
    PROVENANCE_ASSET,
    PUBLISH_LOCK,
    export_dir,
    export_key,
    is_report_notebook,
    is_stale,
    load_pins,
    report_notebooks,
    save_pins,
    set_provenance,
)

ROOT = Path(__file__).parent.parent.resolve()
DOCS = ROOT / "docs"


def notebooks_to_export(paths: list[str]) -> list[Path]:
    """The report notebooks to export — the given ones, or every report under ``docs/``.

    Source-only example notebooks (``# mini:source-only``, e.g. ``docs/gpt.py``) are skipped even when named explicitly: the site links to their GitHub source rather than running them, so exporting one (which re-runs its inline compute) is never intended.
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


def bundle_is_stale(nb: Path) -> bool:
    """Whether *nb*'s bundle is missing or older than anything it's built from."""
    return is_stale(nb, export_dir(nb) / "index.html")


def export_one(nb: Path) -> Path:
    """Export *nb* to ``.mini/exports/<key>/index.html`` (assets land beside it). Returns the dir."""
    out = export_dir(nb) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The render rewrites every asset it still produces, and the sync mirrors whatever is
    # here; clear the last export's first so a figure the report no longer draws (or a
    # ref it stopped reading, via the provenance sidecar) can't ride along as an orphan.
    assets = out.parent / "_assets"
    shutil.rmtree(assets, ignore_errors=True)
    sidecar = assets / PROVENANCE_ASSET
    print(f"  export {nb.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
    # Mark the render as an export so the report's setup cell points its publisher at
    # the bundle (_assets/, beside this HTML); under interactive `marimo edit` the var is
    # absent and figures go to the notebook's public/ instead — see mini.reports.exporting.
    env = {**os.environ, EXPORTING_ENV: "1"}
    subprocess.run(["marimo", "export", "html", "-f", str(nb), "-o", str(out)], check=True, cwd=ROOT, env=env)
    clean_html(out)  # scrub terminal control seqs + redact modal URLs from the published HTML
    default_hidden_code(out)  # literate reports open with code collapsed; the menu toggle still reveals it
    if sidecar.exists():  # the render read store refs — cite their producers in a footer
        refs = json.loads(sidecar.read_text()).get("refs", {})
        out.write_text(set_provenance(out.read_text("utf-8"), refs), "utf-8")
    return out.parent


def publish_one(nb: Path, store) -> str | None:
    """Export *nb*, mirror its bundle to ``exports/<key>/``, and return its revision.

    The revision (a publish-tier commit sha, ``None`` on a history-less bucket) is what the caller pins in ``docs/publish.lock`` — the site serves the bundle at that exact commit, so this publish changes nothing deployed until the pin lands on main.
    """
    bundle = export_one(nb)
    key = export_key(nb)
    print(f"  sync   {bundle.relative_to(ROOT)} -> exports/{key}/")
    return store.sync_export(bundle, key)


def update_pins(new: dict[str, str]) -> None:
    """Fold this run's pins into ``docs/publish.lock``, pruning keys with no notebook.

    Pruning uses the *full* report set (not just what was published now), so a partial publish never drops other reports' pins, but a deleted notebook's pin doesn't linger. The manifest must be committed for the pins to take effect — it's the identity half of a publish; the upload was only evidence.
    """
    live = {export_key(nb) for nb in report_notebooks(DOCS)}
    pins = {k: v for k, v in (load_pins(ROOT) | new).items() if k in live}
    save_pins(ROOT, pins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true", help="mirror each bundle to the HF publish tier after exporting")
    ap.add_argument("--all", action="store_true", help="with --publish: explicitly publish every report under docs/")
    ap.add_argument(
        "--stale-only",
        action="store_true",
        help="skip reports whose bundle is newer than the notebook (mtime heuristic)",
    )
    ap.add_argument("notebooks", nargs="*", help="report notebooks (default: all under docs/)")
    args = ap.parse_args()

    if args.publish and args.stale_only:
        ap.error("--stale-only is a preview optimization; publishing always re-exports")
    if args.publish and not args.notebooks and not args.all:
        ap.error("refusing to publish every report implicitly — name the notebooks, or pass --all")

    nbs = notebooks_to_export(args.notebooks)
    if not nbs:
        sys.exit("No report notebooks found under docs/.")

    if not args.publish:
        if args.stale_only:
            for nb in (fresh := [nb for nb in nbs if not bundle_is_stale(nb)]):
                print(f"  fresh  {nb.relative_to(ROOT)} (bundle newer than notebook — `--force` re-exports)")
            nbs = [nb for nb in nbs if nb not in fresh]
        for nb in nbs:
            export_one(nb)
        print(f"\n{len(nbs)} bundle(s) exported to .mini/exports/." if nbs else "\nNothing stale; bundles untouched.")
        return

    publish_all(nbs)


def publish_all(nbs: list[Path]) -> None:
    """Publish each notebook's bundle, then pin the revisions in ``docs/publish.lock``."""
    from mini.hf_store import HFStore
    from mini.store import store_for

    store = store_for(ROOT / ".mini" / "store")
    if not isinstance(store, HFStore):
        sys.exit("No HF bucket configured — set [tool.mini] store-bucket and run `./go auth`, then retry --publish.")
    pins = {}
    for nb in nbs:
        if (rev := publish_one(nb, store)) is not None:
            pins[export_key(nb)] = rev
    target = store.publish_repo or store.bucket  # exports route to the repo when a publish tier is set (#38)
    print(f"\nPublished {len(nbs)} report(s) to {target}.")
    if pins:
        update_pins(pins)
        print(
            f"Pinned in {PUBLISH_LOCK}: " + ", ".join(f"{k} @ {v[:12]}" for k, v in sorted(pins.items())) + "\n"
            "Commit the lock file — the site serves each report at its pinned revision, so\n"
            "nothing deployed changes until the pin lands on main (PR previews use the branch's)."
        )
    else:
        print('Trigger the Pages build to update the site (push to main, or run the "Deploy Docs" workflow).')


if __name__ == "__main__":
    main()
