# Restoring from this backup

Three legs, each restorable on its own. Nothing here needs a token beyond write access to wherever you are restoring *to*; the backup itself is readable with whatever access this repo and the backup dataset grant.

The names below are the template's. Substitute the four values from the `env:` block of [`.github/workflows/backup.yml`](./.github/workflows/backup.yml).

## Code

The `mirror` branch tracks the source's `main`, and each `snap/<date>` tag is the tip as of that night. To restore `main` into a fresh (or emptied) repo:

```bash
git clone --branch mirror https://github.com/z0u/mi-ni-backup restored
cd restored
git remote add target https://github.com/<owner>/<repo>.git
git push target mirror:main
git push target 'refs/tags/source/*:refs/tags/*'   # the source's own tags, if wanted
```

To restore to a particular night instead, push `snap/<date>` rather than `mirror`. If the mirror stopped advancing at some point (the workflow warns when the source's history was rewritten), the snapshots still hold each night's tip.

## Store bucket

`store/` in the backup dataset is a copy of the bucket's tree: `cas/` (content-addressed blobs), `refs/` (name → artifact pointers, as of the last run), and whatever else the bucket held. Into a fresh bucket:

```python
from huggingface_hub import HfApi, snapshot_download

local = snapshot_download("z0u/mi-ni-backup", repo_type="dataset", allow_patterns="store/**")
api = HfApi()  # a token with write on the new bucket
api.sync_bucket(source=f"{local}/store", dest="hf://buckets/<namespace>/<new-bucket>")
```

A blob's path is its sha256, so the copy is verifiable: hash any `cas/<ab>/<sha>` file and compare. `refs/` is the head's copy; an earlier pointer is in the backup's history (`snapshot_download(..., revision=<backup commit>)`).

## Publish repo

`pub/` replays the source's history commit by commit. Each backup commit titled `pub: replay <source sha>` holds the source at that sha (plus any file a later source commit deleted, which the backup keeps), and `pub/SOURCE_COMMIT` names the source sha the head corresponds to.

To restore the latest state into a fresh dataset repo:

```python
from huggingface_hub import HfApi, snapshot_download

local = snapshot_download("z0u/mi-ni-backup", repo_type="dataset", allow_patterns="pub/**")
api = HfApi()  # write on the new repo
api.upload_folder(repo_id="<namespace>/<new-pub>", repo_type="dataset", folder_path=f"{local}/pub")
```

To recover one pinned revision, which is what a `publish.lock` entry needs: find the backup commit whose title names that source sha (`api.list_repo_commits("z0u/mi-ni-backup", repo_type="dataset")`), and `snapshot_download(..., revision=<backup commit>, allow_patterns="pub/exports/<key>/**")`. Re-publishing it mints a new sha on the new repo, so update the lock file to match.

## Proving it works

Once a quarter, restore each leg into a throwaway target and compare: `git diff mirror <restored main>` for the code, a hash walk over `store/cas/` for the bucket, and the current `publish.lock` keys resolving on the restored publish repo. The mi-ni template's `storage-envs` skill describes a dev pair of repos; seeding it from this backup is a restore drill with a use.
