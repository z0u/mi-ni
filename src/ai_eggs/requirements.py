import re
import subprocess
from typing import overload


@overload
def freeze(*packages: str, dev: bool) -> list[str]: ...


@overload
def freeze(*, all: bool, dev: bool) -> list[str]: ...


def freeze(*packages, all: bool = False, dev: bool = False) -> list[str]:
    """
    Get the requirements for specified packages and their dependencies.

    Args:
        *packages: Names of packages to get requirements for.
        all: If True, freeze all packages. Packages must not be specified if this is True.
        dev: If True, include development dependencies.

    Returns:
        A list of package specifications in the format 'package==version'.
    """
    cmd = ["uv", "tree"]

    if all:
        if packages:
            raise ValueError("Cannot specify packages when 'all' is True")
    else:
        if not packages:
            return []
        cmd.extend([f"--package={pkg}" for pkg in packages])

    if not dev:
        cmd.append("--no-dev")

    result = subprocess.check_output(cmd, text=True)

    # When 'all' is True, ignore the first package since it is the project itself.
    return parse_uv_tree_output(result, ignore_first=all)


def parse_uv_tree_output(output: str, ignore_first: bool) -> list[str]:
    """Parse the output of 'uv tree' command to extract package specifications."""
    requirements: set[str] = set()

    lines = output.strip().split('\n')
    if ignore_first:
        lines = lines[1:]

    # Regular expression to extract package name and version
    # This matches lines like "package v1.2.3" with or without tree characters
    # https://packaging.python.org/en/latest/specifications/name-normalization/#name-format
    name_pattern = r'([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])'
    pattern = name_pattern + r' v([^\s]+)'

    for line in lines:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            pkg_name = match.group(1)
            version = match.group(2)
            requirements.add(f"{pkg_name}=={version}")

    return sorted(requirements)
