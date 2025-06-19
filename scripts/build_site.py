#!/usr/bin/env python
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import argparse

# --- Configuration ---
# Use pathlib for robust path handling
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent
SITE_DIR = WORKSPACE_ROOT / '_site'
DOCS_DIR = WORKSPACE_ROOT / 'docs'
CSS_FILE = SCRIPT_DIR / 'build-site-nb.css'
NBCONVERT_CONFIG = SCRIPT_DIR / 'build-site-config.py'
README_FILE = WORKSPACE_ROOT / 'README.md'
CSS_MARKER = '/* Custom styles for prose width */'  # Used to check if CSS is already injected

# --- Helper Functions ---


def run_command(cmd_list, check=True, cwd=None, capture_output=False):
    """Run a command using subprocess and handles errors."""
    print(f'Running command: {" ".join(map(str, cmd_list))}')
    try:
        result = subprocess.run(
            cmd_list,
            check=check,
            cwd=cwd,
            text=True,
            capture_output=capture_output,
            env=os.environ,
        )
        if capture_output:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f'Error running command: {e}', file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f'Error: Command not found: {cmd_list[0]}', file=sys.stderr)
        sys.exit(1)


# --- Build Steps ---


def prepare_dirs():
    """Remove the existing site directory and creates a new empty one."""
    print('Preparing directories...')
    if SITE_DIR.exists():
        print(f'  Removing existing directory: {SITE_DIR}')
        shutil.rmtree(SITE_DIR)
    print(f'  Creating directory: {SITE_DIR}')
    SITE_DIR.mkdir()


def copy_content():
    """Copy content from docs and the root README into the site directory."""
    print('Copying content...')
    target_readme = SITE_DIR / 'index.md'
    print(f'  Copying {README_FILE} to {target_readme}')
    shutil.copy(README_FILE, target_readme)

    print(f'  Copying contents of {DOCS_DIR} to {SITE_DIR}')
    if DOCS_DIR.is_dir():
        for item in DOCS_DIR.iterdir():
            target_item = SITE_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, target_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_item)
    else:
        print(f'Warning: Docs directory {DOCS_DIR} not found.', file=sys.stderr)


def convert_markdown_to_ipynb():
    """Find Markdown files in _site, converts them to Jupyter notebooks."""
    print('Converting Markdown files to temporary Notebooks...')
    md_files = list(SITE_DIR.rglob('*.md'))
    if not md_files:
        print('  No markdown files found to convert.')
        return

    for md_file in md_files:
        ipynb_file = md_file.with_suffix('.ipynb')
        print(f'  Converting {md_file.relative_to(WORKSPACE_ROOT)} to {ipynb_file.relative_to(WORKSPACE_ROOT)}...')

        try:
            md_content = md_file.read_text(encoding='utf-8').splitlines(keepends=True)

            notebook_json = {
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': md_content}],
                'metadata': {
                    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python'},
                },
                'nbformat': 4,
                'nbformat_minor': 5,
            }

            with open(ipynb_file, 'w', encoding='utf-8') as f:
                json.dump(notebook_json, f, indent=1)

            if not ipynb_file.exists() or ipynb_file.stat().st_size == 0:
                raise IOError(f'Failed to create or write to {ipynb_file}')

            print(f'  Removing original markdown file: {md_file.relative_to(WORKSPACE_ROOT)}')
            md_file.unlink()

        except Exception as e:
            print(f'Error converting {md_file}: {e}', file=sys.stderr)
            sys.exit(1)


def convert_notebooks(no_code: bool = False, output_format: str = 'html'):
    """Convert all .ipynb files in _site using nbconvert."""
    print(f'Converting all Notebooks to {output_format.upper()}...')
    notebook_files = list(SITE_DIR.rglob('*.ipynb'))
    if not notebook_files:
        print('  No notebooks found to convert.')
        return

    cmd = ['uv', 'run', '--', 'jupyter', 'nbconvert', '--config', str(NBCONVERT_CONFIG), '--to', output_format]

    if no_code:
        print('  Excluding code input cells (--no-input).')
        cmd.append('--no-input')

    cmd.extend(map(str, notebook_files))

    run_command(cmd, cwd=WORKSPACE_ROOT)

    print('  Deleting intermediate notebook files...')
    for ipynb_file in notebook_files:
        try:
            print(f'    Deleting {ipynb_file.relative_to(WORKSPACE_ROOT)}')
            ipynb_file.unlink()
        except OSError as e:
            print(f'Error deleting file {ipynb_file}: {e}', file=sys.stderr)


def fix_links(output_format: str = 'html'):
    """Adjust links in generated output files."""
    if output_format == 'markdown':
        target_ext = '.md'
    else:
        target_ext = '.html'

    print(f'Fixing internal links in {target_ext} files...')
    output_files = list(SITE_DIR.rglob(f'*{target_ext}'))
    if not output_files:
        print(f'  No {target_ext} files found to fix links in.')
        return

    docs_prefix_pattern = re.compile(r'(href|src)="(?:\.?/)?docs/([^"]*)"')
    md_link_prefix_pattern = re.compile(r'\]\((?:\.?/)?docs/([^)]*)\)')
    ipynb_ext_pattern_html = re.compile(r'(href|src)="([^"#?]+)\.(?:ipynb|md)(#|\?|")')
    ipynb_ext_pattern_md = re.compile(r'\]\(([^)#?]+)\.(?:ipynb|md)(#|\?|\))')

    for file_path in output_files:
        print(f'  Processing links in {file_path.relative_to(WORKSPACE_ROOT)}...')
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content

            if output_format == 'html':
                content = docs_prefix_pattern.sub(r'\1="\2"', content)

                def replace_ipynb_html(match):
                    attr_type = match.group(1)
                    base_link = match.group(2)
                    trailing_char = match.group(3)
                    return f'{attr_type}="{base_link}{target_ext}{trailing_char}'

                content = ipynb_ext_pattern_html.sub(replace_ipynb_html, content)
            elif output_format == 'markdown':
                content = md_link_prefix_pattern.sub(r'](\1)', content)

                def replace_ipynb_md(match):
                    base_link = match.group(1)
                    trailing_char = match.group(2)
                    return f']({base_link}{target_ext}{trailing_char}'

                content = ipynb_ext_pattern_md.sub(replace_ipynb_md, content)

            if content != original_content:
                print(f'    Links updated in {file_path.relative_to(WORKSPACE_ROOT)}')
                file_path.write_text(content, encoding='utf-8')
            else:
                print(f'    No link changes needed for {file_path.relative_to(WORKSPACE_ROOT)}')

        except Exception as e:
            print(f'Error processing links in {file_path}: {e}', file=sys.stderr)


def add_nojekyll():
    """Create a .nojekyll file in the site directory."""
    print('Adding .nojekyll file...')
    nojekyll_file = SITE_DIR / '.nojekyll'
    try:
        nojekyll_file.touch()
        print(f'  Created {nojekyll_file}')
    except OSError as e:
        print(f'Error creating {nojekyll_file}: {e}', file=sys.stderr)
        sys.exit(1)


def inject_css(output_format: str = 'html'):
    """Inject custom CSS into the head of HTML files."""
    if output_format != 'html':
        print(f'Skipping CSS injection for {output_format.upper()} format.')
        return

    print('Injecting custom CSS into HTML files...')
    if not CSS_FILE.is_file():
        print(f'Error: CSS file not found at {CSS_FILE}', file=sys.stderr)
        sys.exit(1)

    try:
        css_rules = CSS_FILE.read_text(encoding='utf-8')
        style_block = f'<style>\n{css_rules}\n</style>'
    except Exception as e:
        print(f'Error reading CSS file {CSS_FILE}: {e}', file=sys.stderr)
        sys.exit(1)

    html_files = list(SITE_DIR.rglob('*.html'))
    if not html_files:
        print('  No HTML files found to inject CSS into.')
        return

    head_end_pattern = re.compile(r'</head>', re.IGNORECASE)

    for html_file in html_files:
        print(f'  Processing CSS for {html_file.relative_to(WORKSPACE_ROOT)}...')
        try:
            content = html_file.read_text(encoding='utf-8')

            if CSS_MARKER in content:
                print(f'    Skipping {html_file.relative_to(WORKSPACE_ROOT)}, custom CSS already present.')
                continue

            match = head_end_pattern.search(content)
            if match:
                insert_pos = match.start()
                new_content = content[:insert_pos] + '\n' + style_block + '\n' + content[insert_pos:]
                print(f'    Injecting CSS into {html_file.relative_to(WORKSPACE_ROOT)}')
                html_file.write_text(new_content, encoding='utf-8')
            else:
                print(
                    f'    Warning: Could not find </head> tag in {html_file.relative_to(WORKSPACE_ROOT)}. CSS not injected.',
                    file=sys.stderr,
                )

        except Exception as e:
            print(f'Error injecting CSS into {html_file}: {e}', file=sys.stderr)

    print('CSS injection complete.')


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(description='Build the project website.')
    parser.add_argument(
        '--no-code',
        action='store_true',
        help='Exclude code input cells from the generated output files.',
    )
    parser.add_argument(
        '--format',
        choices=['html', 'markdown'],
        default='html',
        help='Output format for converted notebooks (default: html).',
    )
    args = parser.parse_args()

    prepare_dirs()
    copy_content()
    convert_markdown_to_ipynb()
    convert_notebooks(no_code=args.no_code, output_format=args.format)
    fix_links(output_format=args.format)
    add_nojekyll()
    inject_css(output_format=args.format)
    print(f'\nSite build complete ({args.format.upper()} format). Wrote to {SITE_DIR.relative_to(os.getcwd())}/')
    if args.format == 'html':
        print('You can now serve the site using a simple HTTP server.')
        print('For example, using Python 3:')
        print(f'\n    python -m http.server 8000 -d {SITE_DIR.relative_to(os.getcwd())}\n')


if __name__ == '__main__':
    main()
