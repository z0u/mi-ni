# Tagging v1.0.0

This PR adds the necessary tooling to tag the main branch as `v1.0.0`.

## Option 1: Using the shell script (Recommended)

After merging this PR to main, run the following script:

```bash
./scripts/create-v1.0.0-tag.sh
```

This script will:
1. Fetch the latest changes from the main branch
2. Create an annotated tag `v1.0.0` on main
3. Push the tag to the remote repository

## Option 2: Using GitHub Actions workflow

After merging this PR to main:

1. Go to the [Actions tab](https://github.com/z0u/mi-ni/actions)
2. Select "Create Tag" workflow from the left sidebar
3. Click "Run workflow"
4. Fill in the parameters:
   - Tag name: `v1.0.0`
   - Tag message: `Release v1.0.0`
   - Target branch: `main`
5. Click "Run workflow"

The workflow will create and push the tag automatically.

## Option 3: Manual tagging

If you prefer to create the tag manually:

```bash
# Checkout main and ensure it's up to date
git checkout main
git pull origin main

# Create an annotated tag
git tag -a v1.0.0 -m "Release v1.0.0"

# Push the tag
git push origin v1.0.0
```

## Verification

After creating the tag, verify it was created successfully:

```bash
git tag -l -n9 v1.0.0
```

Or view it on GitHub:
https://github.com/z0u/mi-ni/releases/tag/v1.0.0
