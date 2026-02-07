#!/usr/bin/env bash
# Script to create and push tag v1.0.0 on main branch
# This script should be run manually after merging this PR

set -e

TAG_NAME="v1.0.0"
TAG_MESSAGE="Release v1.0.0"
TARGET_BRANCH="main"

echo "Creating tag $TAG_NAME on $TARGET_BRANCH..."

# Fetch the latest changes
git fetch origin

# Checkout the target branch
git checkout "$TARGET_BRANCH"
git pull origin "$TARGET_BRANCH"

# Create the annotated tag
git tag -a "$TAG_NAME" -m "$TAG_MESSAGE"

# Push the tag to remote
git push origin "$TAG_NAME"

echo "✓ Tag $TAG_NAME created and pushed successfully!"
echo ""
echo "View the tag at: https://github.com/z0u/mi-ni/releases/tag/$TAG_NAME"
