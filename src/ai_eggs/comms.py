import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeAlias, TypeVar

import modal

T = TypeVar('T')
Handler: TypeAlias = Callable[[T], None]

@asynccontextmanager
async def send_to(receive: Handler[T], trailing_timout=5) -> AsyncGenerator[Handler[T]]:
    """
    Simple communication channel between local and remote code.

    Args:
        receive: A callback function to receive messages

    Yields:
        A function that sends messages to `receive`
    """
    async with modal.Queue.ephemeral() as q:
        def send(value: T) -> None:
            q.put(('message', value))

        stop_event = asyncio.Event()

        async def consume() -> None:
            # First phase: consume while producer is running
            while not stop_event.is_set():
                get_task = asyncio.create_task(q.get.aio())
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    [get_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    key, value = get_task.result()
                    if key == 'message':
                        receive(value)

            # Second phase: consume remaining messages with timeout
            try:
                async with asyncio.timeout(trailing_timout):
                    while True:
                        key, value = await q.get.aio()
                        if key == 'message':
                            receive(value)
                        elif key == 'sentinel':
                            break
            except TimeoutError:
                pass

        task = asyncio.create_task(consume())
        try:
            yield send
        finally:
            q.put(('sentinel', None))
            stop_event.set()
            await task
