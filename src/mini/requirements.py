import logging
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Callable

import modal

log = logging.getLogger(__name__)


def uv_freeze(
    *packages: str,
    groups: str | list[str] | None = None,
    not_groups: str | list[str] | None = None,
    only_groups: str | list[str] | None = None,
    all_groups: bool = False,
    indexes: str | list[str] | None = None,
    python_version: str | None = None,
    python_platform: str | None = None,
    only_run_locally: bool = True,
) -> list[str]:
    groups = [groups] if isinstance(groups, str) else groups
    not_groups = [not_groups] if isinstance(not_groups, str) else not_groups
    only_groups = [only_groups] if isinstance(only_groups, str) else only_groups
    indexes = [indexes] if isinstance(indexes, str) else indexes

    if only_run_locally and not modal.is_local():
        log.info("Skipping package-freezing: not running locally")
        return []

    cmd = ["uv", "--offline", "tree"]

    result = subprocess.run(cmd + ["--no-dedupe", "--all-groups"], text=True, capture_output=True, check=True)
    all_deps = parse_uv_tree_output(result.stdout, ignore_first=True)

    opts: list[str | tuple[str, ...]] = []
    opts += [("--package", pkg) for pkg in packages]
    opts += [("--group", g) for g in (groups or [])]
    opts += [("--no-group", g) for g in (not_groups or [])]
    opts += [("--only-group", g) for g in (only_groups or [])]
    opts += [("--index", i) for i in (indexes or [])]
    if all_groups:
        opts += ["--all-groups"]
    if python_version:
        opts += [("--python-version", python_version)]
    if python_platform:
        opts += [("--python-platform", python_platform)]

    # Flatten tuples
    opts = [(opt,) if isinstance(opt, str) else opt for opt in opts]
    flat_opts = [opt for sublist in opts for opt in sublist]

    result = subprocess.run(cmd + flat_opts, text=True, capture_output=True, check=True)
    selected_deps = parse_uv_tree_output(result.stdout, ignore_first=True)
    log.info(f"Selected {len(selected_deps)} of {len(all_deps)} dependencies")
    log.debug("Dependencies: %s", selected_deps)
    return selected_deps


def parse_uv_tree_output(output: str, ignore_first: bool) -> list[str]:
    """Parse the output of 'uv tree' command to extract package specifications."""
    requirements: set[str] = set()

    lines = output.strip().split("\n")
    if ignore_first:
        lines = lines[1:]

    # Regular expression to extract package name, optional extras, and version.
    # Matches lines like "package v1.2.3" and "package[extra] v1.2.3", with or
    # without tree characters.
    # https://packaging.python.org/en/latest/specifications/name-normalization/#name-format
    name_pattern = r"([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])"
    extras_pattern = r"(\[[A-Z0-9._,-]+\])?"
    pattern = name_pattern + extras_pattern + r" v([^\s]+)"

    for line in lines:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            pkg_name = match.group(1)
            extras = match.group(2) or ""
            version = match.group(3)
            # Strip local version identifier (e.g., +cpu, +cu121) for cross-platform compatibility
            # Modal and other environments may not have the same local builds available
            version = version.split("+")[0]
            requirements.add(f"{pkg_name}{extras}=={version}")

    return sorted(requirements)


def _dir_contains_python(path: Path) -> bool:
    """Return True if directory looks like a Python package (has __init__.py or any .py)."""
    if (path / "__init__.py").exists():
        return True
    return any(path.rglob("*.py"))


def _scan_src_packages(root_dir: Path) -> list[str]:
    """Discover top-level packages and modules under src/ (UV namespace discovery)."""
    src_dir = root_dir / "src"
    if not src_dir.is_dir():
        return []
    names: set[str] = set()
    for child in src_dir.iterdir():
        n = child.name
        if n.startswith(".") or n.startswith("_"):
            continue
        if child.is_dir() and _dir_contains_python(child):
            names.add(n)
        elif child.is_file() and child.suffix == ".py" and child.stem != "__init__":
            names.add(child.stem)
    return sorted(names)


def _packages_from_uv_config(pyproject: dict, root_dir: Path) -> list[str]:
    """Load explicit packages from tool.uv.build-backend.packages if provided."""
    backend = pyproject.get("tool", {}).get("uv", {}).get("build-backend", {})
    pkgs = backend.get("packages") if isinstance(backend, dict) else None
    if not isinstance(pkgs, list):
        return []
    names: set[str] = set()
    for entry in pkgs:
        if not isinstance(entry, str):
            continue
        # try as relative to project root
        path = root_dir / entry
        if path.is_dir():
            names.add(path.name)
            continue
        if path.is_file() and path.suffix == ".py" and path.stem != "__init__":
            names.add(path.stem)
            continue
        # try relative to src/
        src_path = root_dir / "src" / entry
        if src_path.is_dir():
            names.add(src_path.name)
        elif src_path.is_file() and src_path.suffix == ".py" and src_path.stem != "__init__":
            names.add(src_path.stem)
    return sorted(names)


def _packages_from_hatch(pyproject: dict, root_dir: Path) -> list[str]:
    """Load packages from Hatch config if present (legacy fallback)."""
    hatch_packages = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    if not isinstance(hatch_packages, list):
        return []
    paths = [root_dir / d for d in hatch_packages if isinstance(d, str)]
    directories = [path for path in paths if path.is_dir()]
    return sorted(path.name for path in directories)


def project_packages() -> list[str]:
    """
    Determine first-party package/module names for this repo.

    Strategy (in order):
    - If using UV build backend with `namespace = true`, scan `src/` for top-level
      Python packages (directories) and modules (single .py files). This mirrors UV's
      auto-discovery behavior and avoids coupling to a specific backend.
    - Else, fall back to Hatch's `tool.hatch.build.targets.wheel.packages` if present.

    Returns:
        Sorted list of top-level import names (e.g., ['ex_color', 'infra']).
    """
    root_dir = find_project_root()

    pyproject_path = root_dir / "pyproject.toml"
    log.debug(f"Loading {pyproject_path}")
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    # 1) Prefer UV build-backend with namespace discovery: scan src/
    uv_backend = pyproject.get("tool", {}).get("uv", {}).get("build-backend", {})
    namespace_enabled = bool(uv_backend.get("namespace", False)) if isinstance(uv_backend, dict) else False

    strategies: list[tuple[str, Callable[[], list[str]]]] = []
    if namespace_enabled:
        strategies.append(("src scan", lambda: _scan_src_packages(root_dir)))
    # Try explicit UV packages if provided
    strategies.append(("uv config", lambda: _packages_from_uv_config(pyproject, root_dir)))
    # Fallback to Hatch config
    strategies.append(("hatch config", lambda: _packages_from_hatch(pyproject, root_dir)))

    for label, fn in strategies:
        packages = fn()
        if packages:
            log.info(f"Found {len(packages)} local packages via {label}: {', '.join(packages)}")
            log.debug("Packages: %s", packages)
            return packages

    log.info("No recognizable package configuration; returning empty list")
    return []


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
        if (current / "pyproject.toml").exists():
            return current

        parent = current.parent
        if parent == current:  # Reached the file system root
            break
        current = parent

    raise FileNotFoundError(f"Could not find pyproject.toml from {current}")
