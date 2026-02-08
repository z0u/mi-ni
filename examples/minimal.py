import sys
import time
from typing import Literal
from mini import LocalExecutor, ModalExecutor, emit_progress


def train(x: int) -> int:
    """A task that reports progress."""
    for i in range(10):
        time.sleep(0.1)
        emit_progress(i + 1, 10, message=f'processing item {x}')
    return x * 2


def main(loc: Literal['local', 'modal']):
    if loc == 'local':
        executor = LocalExecutor('local', max_workers=3)
    else:
        executor = ModalExecutor('modal').w(timeout=60, max_containers=3)
    print(f'Using executor: {executor}')
    results = list(executor.map(train, [1, 2, 3, 4, 5]))
    print('Results:', results)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'local'
    assert mode in ('local', 'modal'), f'Unknown mode: {mode}'
    main(mode)
