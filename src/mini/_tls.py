"""
Make Modal's gRPC client work behind TLS-inspecting proxies.

Modal talks to its control plane over gRPC (``grpclib``), and ``grpclib`` builds
its SSL context from **certifi's** CA bundle alone — it passes ``cafile=`` to
``ssl.create_default_context``, which *replaces* the system trust store rather
than adding to it. In environments that route egress through a TLS-inspecting
proxy (many corporate networks, and the Claude Code sandbox), the proxy's CA is
installed in the *system* bundle (``SSL_CERT_FILE`` /
``/etc/ssl/certs/ca-certificates.crt``) but not in certifi's — so Modal's
handshake fails with ``CERTIFICATE_VERIFY_FAILED: self-signed certificate in
certificate chain`` even though ``pip``/``requests`` work fine.

The fix: build a combined bundle (certifi **plus** the system CAs) and point
``certifi.where()`` at it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import tempfile
from pathlib import Path

__all__ = ["ensure_grpc_trusts_system_ca"]

log = logging.getLogger(__name__)

_configured = False


def _system_ca_files() -> list[str]:
    """Candidate system CA bundles, most-specific first (env wins over defaults)."""
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    return [c for c in candidates if c]


def ensure_grpc_trusts_system_ca() -> None:
    """Point ``certifi.where()`` at a certifi+system combined CA bundle.

    Idempotent and best-effort: any failure leaves certifi untouched. Safe to
    call before any Modal connection (e.g. from ``ModalApparatus.__init__``).
    """
    global _configured
    if _configured:
        return
    _configured = True

    try:
        import certifi
    except ImportError:
        return

    certifi_path = Path(certifi.where())
    try:
        base = certifi_path.read_bytes()
    except OSError:
        return

    extra = bytearray()
    seen = {os.path.realpath(certifi_path)}
    for cand in _system_ca_files():
        real = os.path.realpath(cand)
        if real in seen:
            continue
        seen.add(real)
        try:
            extra += b"\n" + Path(cand).read_bytes()
        except OSError:
            continue

    if not extra.strip():
        return  # nothing beyond certifi to add — leave certifi as-is

    combined = base + bytes(extra)
    digest = hashlib.sha256(combined).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / f"mini-ca-bundle-{digest}.pem"
    try:
        if not out.exists():
            out.write_bytes(combined)
    except OSError:
        return

    certifi.where = lambda: str(out)  # type: ignore
    log.debug("Pointed certifi at combined CA bundle for gRPC TLS: %s", out)
