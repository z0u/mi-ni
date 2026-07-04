# Operational constraints worth remembering

*Part of the [engineering notes](./README.md).*

- **Egress allow-list.** Bucket I/O needs `*.xethub.hf.co`; serving figures needs
  `*.cdn.hf.co`. Without them, metadata calls to `huggingface.co` succeed while every
  byte transfer hangs on a 403 — a confusing failure mode.
- **Modal gRPC TLS.** Modal's client builds its trust store from `certifi` alone; behind
  a TLS-inspecting proxy it needs the system CA folded in
  (`mini._tls.ensure_grpc_trusts_system_ca` already does this).
- **CORS / Range.** The bucket reflects the request `Origin` on both the resolve redirect
  and the CDN response, and the CDN advertises `Accept-Ranges` — so a Pages-served report
  can `fetch()` a published JSON cross-origin and Range-slice a big binary.
