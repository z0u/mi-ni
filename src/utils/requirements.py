import logging
import re
import subprocess
import tomllib
from pathlib import Path
from typing import overload

log = logging.getLogger(__name__)


@overload
def freeze(*packages: str, dev: bool = False, local: bool = False) -> list[str]: ...


@overload
def freeze(*, all: bool, dev: bool = False, local: bool = False) -> list[str]: ...


def freeze(*packages, all: bool = False, dev: bool = False, local: bool = False) -> list[str]:
    """
    Get the requirements for specified packages and their dependencies.

    Args:
        *packages: Names of packages to get requirements for.
        all: If True, freeze all packages. Packages must not be specified if this is True.
        dev: If True, include development dependencies.
        local: If True, include dependencies from the 'local' group.

    Returns:
        A list of package specifications in the format 'package==version'.

    """
    cmd = ['uv', '--offline', 'tree']

    if all:
        if packages:
            raise ValueError("Cannot specify packages when 'all' is True")
        package_opts = []
    else:
        if not packages:
            return []
        package_opts = [f'--package={pkg}' for pkg in packages]
        cmd.extend([f'--package={pkg}' for pkg in packages])

    result = subprocess.run(cmd + ['--no-dedupe', '--all-groups'], text=True, capture_output=True, check=True)
    available_deps = parse_uv_tree_output(result.stdout, ignore_first=True)

    constraints = ['--no-dedupe', '--all-groups']
    if not dev:
        constraints.append('--no-dev')
    if not local:
        constraints.append('--no-group=local')

    result = subprocess.run(cmd + constraints + package_opts, text=True, capture_output=True, check=True)
    selected_deps = parse_uv_tree_output(result.stdout, ignore_first=all)

    log.info(f'Selected {len(selected_deps)} of {len(available_deps)} dependencies')
    log.debug('Dependencies: %s', selected_deps)
    return selected_deps


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
            requirements.add(f'{pkg_name}=={version}')

    return sorted(requirements)


def project_packages() -> list[str]:
    """
    Find and return local packages defined in pyproject.toml.

    Returns:
        List of local packages as defined in tool.hatch.build.targets.wheel.packages.
    """
    root_dir = find_project_root()

    pyproject_path = root_dir / 'pyproject.toml'
    log.debug(f'Loading {pyproject_path}')
    with open(pyproject_path, 'rb') as f:
        pyproject = tomllib.load(f)

    log.debug('Using tool.hatch.build.targets.wheel.packages')
    paths = [
        Path(d)
        for d in pyproject.get('tool', {})
        .get('hatch', {})
        .get('build', {})
        .get('targets', {})
        .get('wheel', {})
        .get('packages', [])
    ]
    directories = [path for path in paths if path.is_dir()]
    packages = [path.name for path in directories]

    log.info(f'Found {len(packages)} local packages in the project')
    log.debug('Packages: %s', packages)
    return packages


def find_project_root() -> Path:
    """
    Find the project root directory containing pyproject.toml.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found.
    """
    current = Path.cwd()

    # Try a few times going up the directory tree
    for _ in range(10):
        if (current / 'pyproject.toml').exists():
            return current

        parent = current.parent
        if parent == current:  # Reached the file system root
            break
        current = parent

    raise FileNotFoundError('Could not find pyproject.toml')
