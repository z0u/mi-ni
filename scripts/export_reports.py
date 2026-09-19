#!/usr/bin/env python
"""Export reports to self-contained bundles, optionally syncing to the bucket.

Each report (a literate script under ``docs/``; :func:`~mini.reports.is_report`) exports to its own bundle at ``.mini/exports/<key>/`` — ``index.html`` plus the name-keyed ``_assets/`` its publisher wrote (``mini.lit.render`` installs a publisher aimed at the output's ``_assets/``). After the weave, provenance, thumbnails, the PDF and the sync follow. With ``--publish`` each bundle is then mirrored to the configured HF bucket at ``exports/<key>/``: the authenticated half of publishing (it needs the data the report reads + a write token). ``scripts/build_site.py`` assembles the site from these bundles — the synced ones in CI (read-only), the local ones offline.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `import build_site` (sibling) works

from build_site import LinkResolver, resolve_html_links  # noqa: E402
from mini.lit import render  # noqa: E402
from mini.report_print import print_bundle  # noqa: E402
from mini.reports import (  # noqa: E402
    MD_LEAF,
    MD_TYPE,
    PDF_LEAF,
    PDF_TYPE,
    PROVENANCE_ASSET,
    export_dir,
    export_key,
    is_report,
    is_stale,
    load_pins,
    publish_lock,
    reports,
    save_pins,
    set_alternate,
    set_provenance,
    set_report_styles,
    write_thumbnails,
)
from mini.store import active_profile  # noqa: E402

ROOT = Path(__file__).parent.parent.resolve()
DOCS = ROOT / "docs"
REPORT_CSS = DOCS / "report.css"


def reports_to_export(paths: list[str]) -> list[Path]:
    """The reports to export — the given ones, or every report under ``docs/``.

    Source-only example scripts (``# mini:source-only``, e.g. ``docs/gpt.py``) are skipped even when named explicitly: the site links to their GitHub source rather than running them, so exporting one (which re-runs its inline compute) is never intended.
    """
    if not paths:
        return reports(DOCS)
    keep = []
    for p in paths:
        path = Path(p).resolve()
        if is_report(path):
            keep.append(path)
        else:
            print(f"  skip {path.name}: source-only example, not a rendered report — see `./go render`")
    return keep


def bundle_is_stale(path: Path) -> bool:
    """Whether *path*'s bundle is missing or older than anything it's built from."""
    return is_stale(path, export_dir(path) / "index.html")


def export_one(path: Path) -> Path:
    """Export *path* to ``.mini/exports/<key>/index.html`` (assets land beside it). Returns the dir."""
    out = export_dir(path) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The render rewrites every asset it still produces, and the sync mirrors whatever is
    # here; clear the last export's first so a figure the report no longer draws (or a
    # ref it stopped reading, via the provenance sidecar) can't ride along as an orphan.
    assets = out.parent / "_assets"
    shutil.rmtree(assets, ignore_errors=True)
    sidecar = assets / PROVENANCE_ASSET
    print(f"  export {path.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
    html = _weave(path, out)
    if sidecar.exists():  # the render read store refs — cite their producers in a footer
        refs = json.loads(sidecar.read_text()).get("refs", {})
        html = set_provenance(html, refs)
    # Small copies of every figure, for the index's strips: made here because this is the
    # one step that holds the figure bytes (the site build fetches only the HTML).
    html, thumbs = write_thumbnails(html, assets)
    if thumbs:
        print(f"  thumbs {len(thumbs)} figure(s) -> {assets.relative_to(ROOT)}/thumbs/")
    out.write_text(html, "utf-8")
    # The PDF, for reading on paper or e-ink (todo/eng/pdf-exports.md): printed here, the
    # half that holds the bundle, and synced beside the HTML so the site build can link it
    # without a browser of its own. Author links are resolved the way the published page
    # resolves them, in the printed copy only, so the PDF's links work and its bytes are a
    # function of the report alone (left relative, they would carry the loopback port the
    # print served from, and an unchanged report would upload a new file every time).
    pdf = out.parent / PDF_LEAF
    pdf.unlink(missing_ok=True)  # the last export's, which a skipped print must not leave to sync
    from_dir = path.parent.relative_to(DOCS).as_posix()
    printable = resolve_html_links(
        html,
        LinkResolver.discover(),
        from_dir="" if from_dir == "." else from_dir,
        out_dir=export_key(path),
        externalizing=True,
    )
    print(f"  print  {out.relative_to(ROOT)} -> {pdf.relative_to(ROOT)} (headless Chromium; a few seconds)")
    if print_bundle(out.parent, pdf, html=printable) is not None:
        out.write_text(set_alternate(html, type=PDF_TYPE, href=PDF_LEAF), "utf-8")
    return out.parent


def _weave(script: Path, out: Path) -> str:
    """Weave the literate *script* into the bundle holding *out*, and return the page.

    ``mini.lit.render`` writes the page and the woven Markdown beside it, with figures under the bundle's ``_assets/`` through the report publisher, so the sidecar and the thumbnails read from one place. The Markdown is declared as an alternate rendition, the way the PDF is, so a reader (an agent, mostly) can fetch the text of a published report without parsing the page. The shared stylesheet is inlined here for the print below, since a script's page carries only ``mini.lit``'s own; the site build re-inlines the current source on top.

    A cell that raised is a failed export: the page would carry the traceback where a figure should be, and the sync would publish it.
    """
    rendered = render(script, out_dir=out.parent)
    if errors := rendered.woven.errors:
        lines = "\n\n".join(f"cell at line {o.cell.line}:\n{o.error}" for o in errors)
        sys.exit(f"export of {script.relative_to(ROOT)} failed: {len(errors)} cell(s) raised\n{lines}")
    html = set_alternate(rendered.html, type=MD_TYPE, href=MD_LEAF)
    if REPORT_CSS.exists():
        html = set_report_styles(html, REPORT_CSS.read_text("utf-8"))
    out.write_text(html, "utf-8")
    return html


def publish_one(path: Path, store) -> str | None:
    """Export *path*, mirror its bundle to ``exports/<key>/``, and return its revision.

    The revision (a publish-tier commit sha, ``None`` on a history-less bucket) is what the caller pins in ``docs/publish.lock`` — the site serves the bundle at that exact commit, so this publish changes nothing deployed until the pin lands on main.
    """
    bundle = export_one(path)
    key = export_key(path)
    print(f"  sync   {bundle.relative_to(ROOT)} -> exports/{key}/")
    return store.sync_export(bundle, key)


def update_pins(new: dict[str, str]) -> None:
    """Fold this run's pins into the active manifest (:func:`~mini.reports.publish_lock`), pruning keys with no report.

    Pruning uses the *full* report set (not just what was published now), so a partial publish never drops other reports' pins, but a deleted report's pin doesn't linger. The production manifest must be committed for the pins to take effect — it's the identity half of a publish; the upload was only evidence. A profile's manifest is gitignored: dev pins never reach CI.
    """
    live = {export_key(path) for path in reports(DOCS)}
    pins = {k: v for k, v in (load_pins(ROOT) | new).items() if k in live}
    save_pins(ROOT, pins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true", help="mirror each bundle to the HF publish tier after exporting")
    ap.add_argument("--all", action="store_true", help="with --publish: explicitly publish every report under docs/")
    ap.add_argument(
        "--stale-only",
        action="store_true",
        help="skip reports whose bundle is newer than the script (mtime heuristic)",
    )
    ap.add_argument("reports", nargs="*", help="reports (default: all under docs/)")
    args = ap.parse_args()

    if args.publish and args.stale_only:
        ap.error("--stale-only is a preview optimization; publishing always re-exports")
    if args.publish and not args.reports and not args.all:
        ap.error("refusing to publish every report implicitly — name the reports, or pass --all")

    paths = reports_to_export(args.reports)
    if not paths:
        sys.exit("No reports found under docs/.")

    if not args.publish:
        if args.stale_only:
            for path in (fresh := [path for path in paths if not bundle_is_stale(path)]):
                print(f"  fresh  {path.relative_to(ROOT)} (bundle newer than script — `--force` re-exports)")
            paths = [path for path in paths if path not in fresh]
        for path in paths:
            export_one(path)
        print(
            f"\n{len(paths)} bundle(s) exported to .mini/exports/." if paths else "\nNothing stale; bundles untouched."
        )
        return

    publish_all(paths)


def publish_all(paths: list[Path]) -> None:
    """Publish each report's bundle, then pin the revisions in ``docs/publish.lock``."""
    from mini.hf_store import HFStore
    from mini.store import store_for

    store = store_for(ROOT / ".mini" / "store")
    if not isinstance(store, HFStore):
        sys.exit("No HF bucket configured — set [tool.mini] store-bucket and run `./go auth`, then retry --publish.")
    pins = {}
    for path in paths:
        if (rev := publish_one(path, store)) is not None:
            pins[export_key(path)] = rev
    target = store.publish_repo or store.bucket  # exports route to the repo when a publish tier is set (#38)
    profile = active_profile()
    where = f"{target} (profile {profile})" if profile else target
    print(f"\nPublished {len(paths)} report(s) to {where}.")
    if pins:
        update_pins(pins)
        pinned = f"Pinned in {publish_lock()}: " + ", ".join(f"{k} @ {v[:12]}" for k, v in sorted(pins.items()))
        if profile:
            print(
                f"{pinned}\n"
                f"That manifest is gitignored: a publish under profile {profile!r} deploys nothing and moves no\n"
                "production pin. `./go preview` and a local `./go site` read it; CI never does."
            )
        else:
            print(
                f"{pinned}\n"
                "Commit the lock file — the site serves each report at its pinned revision, so\n"
                "nothing deployed changes until the pin lands on main (PR previews use the branch's)."
            )
    else:
        print('Trigger the Pages build to update the site (push to main, or run the "Deploy Docs" workflow).')


if __name__ == "__main__":
    main()
