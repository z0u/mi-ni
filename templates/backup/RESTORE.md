# Restoring from this backup

The backup has three parts, and each one can be restored on its own. Nothing here needs a token beyond write access to the restore target.

Angle-bracketed names are placeholders. The real ones are in the `env:` block of [`.github/workflows/backup.yml`](./.github/workflows/backup.yml).

## Code

The `mirror` branch tracks `main` in the source repo. Each `snap/<date>` tag records the tip of that branch as of that night. To restore `main` into a fresh (or emptied) repo:

```bash
git clone --branch mirror https://github.com/<owner>/<project>-backup restored
cd restored
git remote add target https://github.com/<owner>/<repo>.git
git push target mirror:main
git push target 'refs/tags/source/*:refs/tags/*'   # tags from the source repo, if wanted
```

For a particular night, push `snap/<date>` instead of `mirror`. The mirror can stop advancing if the history of the source repo was rewritten; the run warns when that happens, and the snapshots still hold the tip from each night.

## Store bucket

`store/` in the backup dataset copies the tree of the bucket: `cas/` for content-addressed blobs, `refs/` for the name → artifact pointers as of the last run, and whatever else the bucket held. To restore into a fresh bucket:

```python
from huggingface_hub import HfApi, snapshot_download

local = snapshot_download("<ns>/<project>-backup", repo_type="dataset", allow_patterns="store/**")
api = HfApi()  # a token with write on the new bucket
api.sync_bucket(source=f"{local}/store", dest="hf://buckets/<namespace>/<new-bucket>")
```

The path of a blob is its sha256, so you can check the copy: hash any `cas/<ab>/<sha>` file and compare. The `refs/` you get is the copy at the head. For an earlier pointer, read it out of the backup history with `snapshot_download(..., revision=<backup commit>)`.

## Publish repo

`pub/` replays the history of the source repo, commit by commit. Each backup commit is titled `pub: replay <source sha>` and holds the source as of that sha. It also keeps any file that a later source commit deleted. `pub/SOURCE_COMMIT` names the source sha that the head corresponds to.

To restore the latest state into a fresh dataset repo:

```python
from huggingface_hub import HfApi, snapshot_download

local = snapshot_download("<ns>/<project>-backup", repo_type="dataset", allow_patterns="pub/**")
api = HfApi()  # write on the new repo
api.upload_folder(repo_id="<namespace>/<new-pub>", repo_type="dataset", folder_path=f"{local}/pub")
```

A `publish.lock` entry pins one revision. To recover it, find the backup commit whose title names that source sha with `api.list_repo_commits("<ns>/<project>-backup", repo_type="dataset")`, then call `snapshot_download(..., revision=<backup commit>, allow_patterns="pub/exports/<key>/**")`. Publishing it again mints a new sha on the new repo, so update the lock file to match.

## Checking that it works

Once a quarter, restore each part into a throwaway target and compare. For the code, run `git diff mirror <restored main>`. For the bucket, walk the hashes over `store/cas/`. For the publish repo, check that the current `publish.lock` keys resolve against the restored copy. The `storage-envs` skill in the mi-ni template describes a pair of dev repos; seeding those from this backup gives you a restore drill that also has a use.
