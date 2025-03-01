import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeAlias, TypeVar

import modal

T = TypeVar('T')
Handler: TypeAlias = Callable[[T], None]


@asynccontextmanager
async def send_to(receive: Handler[T]) -> AsyncGenerator[Handler[T]]:
    """
    Simple communication channel between local and remote code.

    Args:
        receive: A callback function to receive messages

    Yields:
        A function that sends messages to `receive`
    """
    async with modal.Queue.ephemeral() as queue:
        def send(value: T) -> None:
            queue.put(value)

        stop_event = asyncio.Event()

        async def consume() -> None:
            while not stop_event.is_set():
                get_task = asyncio.create_task(queue.get.aio())
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    [get_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    receive(get_task.result())

        task = asyncio.create_task(consume())
        try:
            yield send
        finally:
            stop_event.set()
            await task
