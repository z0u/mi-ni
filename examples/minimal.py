from typing import Literal
from mini import LocalExecutor, ModalExecutor


def train(x: int) -> int:
    print(f'Training on {x}...')
    return x**2


def main(loc: Literal['local', 'modal']):
    if loc == 'local':
        executor = LocalExecutor('local', max_workers=1)
    else:
        executor = ModalExecutor('modal').w(timeout=60)
    print(f'Using executor: {executor}')
    results = list(executor.map(train, [1, 2, 3, 4]))
    print('Results:', results)


if __name__ == '__main__':
    main('local')
    main('modal')
