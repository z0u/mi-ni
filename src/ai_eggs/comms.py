from asyncio import create_task, wait_for
from contextlib import asynccontextmanager
from typing import Any, Callable

import modal


class Comms:
    queue: modal.Queue

    def __init__(self, queue: modal.Queue):
        self.queue = queue

    @classmethod
    @asynccontextmanager
    async def create(cls):
        with modal.Queue.ephemeral() as queue:
            yield cls(queue)

    @asynccontextmanager
    async def auto_close(self):
        try:
            yield
        finally:
            self.close()

    def close(self):
        self.queue.put(None)

    def emit(self, value: Any):
        self.queue.put(value)

    @asynccontextmanager
    async def subscribe(self, receive: Callable[[Any], None]):
        async def consume():
            while message := await self.queue.get.aio():
                if message is None:
                    break
                receive(message)

        task = create_task(consume())
        try:
            yield
        finally:
            task.cancel()
            await wait_for(task, timeout=3)


@asynccontextmanager
async def simple_comms(receive: Callable[[Any], None]):
    async with Comms.create() as comms:
        def send(value):
            comms.emit(value)

        async with comms.subscribe(receive):
            async with comms.auto_close():
                yield send
