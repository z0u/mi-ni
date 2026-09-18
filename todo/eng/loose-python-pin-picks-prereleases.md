---
status: done
tags: [ci, tooling]
opened: 2026-09-11
closed: 2026-09-13
---
# A cached prerelease interpreter can satisfy `.python-version`

`.python-version` holds `3.14`, and `pyproject.toml` asks for `>=3.14`. Both are happy with `3.14.0rc2`, so a container that has rc2 in the uv cache builds its venv on the release candidate and never fetches 3.14.7. That happened in this environment on 2026-09-11: the whole test suite collapsed at import time with `TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'`, because pydantic 2.13.4 calls a `typing._eval_type` signature that only exists in the final release. `uv python install 3.14.7 && uv sync -p 3.14.7` cleared it, and the lock file was unchanged — so the resolution was always correct and only the interpreter was behind.

The failure mode is unpleasant because it looks like a code problem. Forty collection errors with a pydantic traceback reads as "someone broke the models", and the interpreter version appears in one line of pytest's header that is easy to skim past.

Options, roughly in order of how much they cost: pin `.python-version` to a full version and accept editing it a few times a year; raise the floor to `>=3.14.1` in `requires-python`, which excludes every 3.14.0 prerelease and needs no maintenance (a bare `>=3.14` matches prereleases of 3.14.0 under PEP 440, `>=3.14.1` cannot); or have `scripts/install.sh` compare `sys.version_info.releaselevel` against `'final'` and say something when it isn't. The middle one looks cheapest, and CI is unaffected either way since `setup-uv` starts from a clean cache.

## Notes

**2026-09-13, tech debt** — Took the middle option: `requires-python` is now `>=3.14.1`, with the reasoning in a comment beside it. `uv.lock` moved by one line (its own `requires-python`) and no package resolved differently, which is the same "resolution was always correct" signal the incident gave.

Measured the boundary first, on a scratch project with 3.15.0rc2 in the cache, because the premise needs uv's rules rather than PEP 440's — `packaging` puts `3.14.0rc2` *below* `3.14.0`, so under strict ordering a plain `>=3.14` should already have excluded it. uv is deliberately more permissive: it discards prerelease information from a bound inside the same minor series, so `>=3.15`, `>=3.15.0` and even `>=3.15.0.post0` all built the venv on rc2, and `>3.15.0rc2` came back in uv's own error message rewritten as `>3.15.0`. Only `>=3.15.1` refused. So the tempting cheaper variant — `>=3.14.0`, which would have kept 3.14.0 final eligible — does not work, and naming `.1` is the only maintenance-free bound.

That refusal also settles the third option. uv stops with `The Python request from .python-version resolved to Python 3.15.0rc2, which is incompatible with the project's Python requirement`, which names the interpreter and the requirement in one line — so a `releaselevel` check in `scripts/install.sh` would only restate it, and none was added. Note the stop is an error rather than a fall-through to fetching a compatible interpreter: on a container holding only an rc, `uv sync` fails and someone runs `uv python install`. That is the intended outcome here, since the alternative was forty pydantic collection errors that read as a code problem.
