---
status: done
tags: [storage, tooling, security]
opened: 2026-09-05
closed: 2026-09-05
bundle: backup-template
---
# The backup's store leg can't copy from a private source bucket

sca2's first live run of the backup job failed its store leg with a bare `422 Unprocessable Entity` from `POST /api/buckets/z0u-bot/sca2-backup-store/batch`, and copied none of the 1954 files. The publish and code legs were unaffected.

The cause is a mismatch between the credential split and the shape of the server-side copy. The copy is one request: it goes to the *destination* bucket's endpoint carrying the destination's token, and names the source bucket in each `copyFile` operation. The Hub authorises it against that one token — ["You need read access to the source repository or bucket and write access to the destination bucket"](https://huggingface.co/docs/hub/en/storage-buckets#copying-files-between-repos-and-buckets). But the backup design issues no token holding both: `SOURCE_HF_TOKEN` reads the source, and `HF_BUCKET_TOKEN` is minted from the trusted publisher scoped to `buckets/<ns>/<project>-backup-store`. A fine-grained token carries implicit read on public repos only, so a private source is invisible to it and the Hub answers `Source repository not found: buckets/<ns>/<project>-store` — the same wording it uses for a bucket that does not exist.

Probed against the live API with a hash that cannot resolve, so nothing was written: a readable source gives `file not found in source repo`, an unreadable private one gives `Source repository not found`, both under 422. A positive control (a real copy within `z0u/sca2-store`, since that token can read and write it) succeeded, so the mechanism itself is sound. The minted destination token is also fine — the Hub accepted and processed the request, it just could not reach the source.

There are three fixes, and the first two are configuration a project can reach for today. Make the source bucket public, and the implicit public read covers it. Or share the source with the backup account and store one fine-grained token with read on the source and write on both targets as `HF_TOKEN`, which the workflow already prefers over minting; that keeps the token inside the backup account and read-only towards the source, at the cost of rotating it. (Whether buckets support per-repo collaborators is unverified.)

The third is the durable one, and it is ours: `templates/backup/backup.py`. `backup_store` already has a download-then-upload path for files the Hub does not track with Xet; make it the fallback for every file when the write client cannot read the source, chosen by one `bucket_info` probe at the top of the leg. That keeps the design's promise that no token spans both accounts. For sca2 it would move 895 MB on the first run and little after — the files are small (median 26 kB, max 74 MB, largest 200 summing to 0.6 GB), so a staged batch of 200 stays well inside a runner's disk.

Fix the diagnostics in the same pass. The batch endpoint returns a per-file reason (`{"success":false,"failed":[{"path":…,"error":…}]}`), and `hf_raise_for_status` discards the body, so `state/last-run.json` recorded only a URL. Reading the response before raising is what turns this class of failure from opaque into obvious.

## Notes

**2026-09-05, setup** — Diagnosed while installing the template into sca2's backup repo ([z0u/sca2#146](https://github.com/z0u/sca2/pull/146)). That project took the first fix — `z0u/sca2-store` is now public, which nothing in it argued against — so its backup runs clean and nothing there waits on this. What stays open is the template's half: it cannot back up a *private* bucket at all, and the run record swallows the reason. Both are worth fixing so the next project isn't asked to choose between a private store and a working backup.

**2026-09-05, Opus** — Done, the third fix as described. `readable()` asks `bucket_info` once before the copy; where the answer is no, every file goes the path the non-Xet files already took. Batches there are bounded by bytes as well as count (`STAGE_BYTES`, 2 GB) since that path now carries whole stores, with a file larger than the budget going alone rather than being refused. `bucket_batch()` keeps the response body, so a refusal reaches `state/last-run.json` with the Hub's own per-file reason.

Tested against the fake API, which grew a `bucket_info` that distinguishes what the *backup bucket's* credential can see from what the source's read token can — that split being the whole subject. Not exercised against a real private bucket: sca2's is public now, and standing up another to test it wasn't worth the setup. The two configuration fixes stay in the skill as the faster options, since the server-side copy is still the one that costs nothing.
