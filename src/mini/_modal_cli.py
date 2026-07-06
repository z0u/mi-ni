"""Entry point for `bin/modal`: apply the TLS fix, then run modal's real CLI.

The `modal` console script is a separate process from ours, so the
monkeypatch in `mini._tls` (see its docstring for the root cause) never gets
a chance to run when you invoke the CLI directly — only `ModalApparatus`
benefits. This shim applies the patch first, then hands off to modal's own
entry point, so `bin/modal` works behind the same TLS-inspecting proxies.
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points

from mini._tls import ensure_grpc_trusts_system_ca


def main() -> None:
    ensure_grpc_trusts_system_ca()  # before any modal import — see mini/_tls.py

    sys.argv[0] = "modal"  # so click/typer help text reads `modal`, not us

    modal_entry_points = [ep for ep in entry_points(group="console_scripts") if ep.name == "modal"]
    if not modal_entry_points:
        raise RuntimeError("No 'modal' console-script entry point found — is the `modal` package installed?")
    sys.exit(modal_entry_points[0].load()())


if __name__ == "__main__":
    main()
