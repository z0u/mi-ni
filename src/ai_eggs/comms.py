import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

import modal


@asynccontextmanager
async def send_to(receive: Callable[[Any], None]) -> AsyncGenerator[Callable[[Any], None], None]:
    """
    Simple communication channel between local and remote code.
    """
    async with modal.Queue.ephemeral() as queue:
        def send(value):
            queue.put(value)

        stop_event = asyncio.Event()

        async def _receive():
            while not stop_event.is_set():
                get_task = asyncio.create_task(queue.get.aio())
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    [get_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    receive(get_task.result())

        task = asyncio.create_task(_receive())
        try:
            yield send
        finally:
            stop_event.set()
            await task
