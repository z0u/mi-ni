#!/usr/bin/env python
"""Convert Jupyter notebooks to Python files for static analysis."""

import argparse
import json
from pathlib import Path
import sys


def ipynb_to_py(notebook_path: Path | str, output_path: Path | str):
    notebook_path = Path(notebook_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'# Generated from {notebook_path}\n\n')
        f.write('# type: ignore\n')
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if source.strip() == '':
                continue
            f.write(f'# Cell {i}\n{source}\n\n')


def find_notebooks(root_dir: Path):
    return [
        nb_path for nb_path in root_dir.glob('**/*.ipynb')
        if nb_path.is_file()
        and '.ipynb_checkpoints' not in str(nb_path)
    ]  # fmt: skip


def collect_conversion_tasks(sources: list[Path], outpath: Path) -> list[tuple[Path, Path]]:
    """Interpret paths like `cp` would."""
    many_sources = len(sources) > 1 or any(source.is_dir() for source in sources)
    if not many_sources:
        return collect_single(sources[0], outpath)

    if outpath.exists() and not outpath.is_dir():
        raise NotADirectoryError(f'Error: multiple inputs given, but {outpath} is not a directory')

    return collect_many(sources, outpath)


def collect_many(sources: list[Path], outpath: Path) -> list[tuple[Path, Path]]:
    tasks = []
    outdir = outpath
    for source in sources:
        if source.is_file():
            output_path = outdir / f'{source.stem}.py'
            tasks.append((source, output_path))
        elif source.is_dir():
            notebooks = find_notebooks(source)
            for nb_path in notebooks:
                # Preserve relative path structure
                rel_path = nb_path.relative_to(source)
                output_path = outdir / rel_path.with_suffix('.py')
                tasks.append((nb_path, output_path))
        else:
            raise FileNotFoundError(f'Warning: Skipping {source} - not a file or directory')

    return tasks


def collect_single(source: Path, outpath: Path) -> list[tuple[Path, Path]]:
    if not outpath.exists():
        if outpath.suffix == '.py':
            # Single input to single output
            return [(source, outpath)]
        else:
            # Assume directory
            return [(source, outpath / f'{source.stem}.py')]
    elif outpath.is_dir():
        # Single input to directory output
        return [(source, outpath / f'{source.stem}.py')]
    elif outpath.is_file():
        # Single input to single output
        return [(source, outpath)]
    else:
        raise FileNotFoundError(f'Warning: Skipping {source} - not a file or directory')


def process_conversion_tasks(tasks: list[tuple[Path, Path]], verbose: bool = False):
    """Process all notebook conversion tasks."""
    if verbose:
        print(f'Converting {len(tasks)} notebook(s) to Python files:', file=sys.stderr)
    for notebook_path, output_path in tasks:
        if verbose:
            print(f'\t{notebook_path} -> {output_path}', file=sys.stderr)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        ipynb_to_py(notebook_path, output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert Jupyter notebooks to Python files.')
    parser.add_argument('sources', nargs='+', type=Path, help='Paths to notebooks or directories')
    parser.add_argument('outpath', type=Path, help='Path to notebook or directory')

    args = parser.parse_args()
    try:
        tasks = collect_conversion_tasks(args.sources, args.outpath)
    except OSError as e:
        print(f'Error: {e}', file=sys.stderr)
        return

    process_conversion_tasks(tasks, verbose=True)


if __name__ == '__main__':
    main()
