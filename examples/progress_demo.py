"""
Example showing progress display with Rich.

Run this to see the progress bar in action:

    python examples/progress_demo.py
"""

from mini.local_executor import LocalExecutor
from mini.progress import emit_progress
import time


def slow_task(x: int) -> int:
    """A task that reports progress."""
    for i in range(10):
        time.sleep(0.1)
        emit_progress(i + 1, 10, message=f'processing item {x}')
    return x * 2


if __name__ == '__main__':
    # Progress bars are shown automatically!
    executor = LocalExecutor('demo', max_workers=3)

    print('Running tasks with automatic Rich progress display...\n')
    results = list(executor.map(slow_task, range(5)))
    print('\nResults:', results)
