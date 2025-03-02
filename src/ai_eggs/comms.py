import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncGenerator, Callable, Literal, TypeAlias, TypeVar

import modal

T = TypeVar('T')
Handler: TypeAlias = Callable[[list[T]], None]

log = logging.getLogger(__name__)


@asynccontextmanager
async def send_to(
    receive: Handler[T],
    trailing_timeout: float | None = 5,
    errors: Literal['throw', 'log'] = 'log'
) -> AsyncGenerator[Handler[T]]:
    # There can be many producers, but only one consumer.

    stop_event = asyncio.Event()

    async with modal.Queue.ephemeral() as q:
        def produce(value: T) -> None:
            """Send values to the consumer"""
            # This function is yielded as the context, so there may be several
            # distributed producers. It gets pickled and sent to remote workers
            # for execution, so we can't use local synchronization mechanisms.
            # All we have is a distributed queue - but we can send signals on a
            # separate control partition of that queue.

            # Emit values.
            # TODO: Allow caller to provide many values at once.
            q.put_many([value])

            # Notify consumer. This is not atomic so it may sometimes result in
            # more than one message on the signal partition, but that's OK: it
            # just means the consumer may occasionally get an empty list of
            # values.
            if q.len(partition='signal') == 0:
                q.put(True, partition='signal')

        async def consume() -> None:
            """Take values from the queue until the context manager exits"""
            # This function is not exposed, so there's exactly one consumer.
            # It always runs locally.

            while True:
                # Wait until values are produced or the context manager exits.
                tasks = [
                    asyncio.create_task(q.get.aio(partition='signal')),
                    asyncio.create_task(stop_event.wait()),
                ]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Either way, get all available messages.
                n = await q.len.aio()
                if n > 0:
                    values = await q.get_many.aio(n, block=False)
                    for value in values:
                        # TODO: Allow receiver to consume many values at once.
                        receive(value)
                elif stop_event.is_set():
                    break

        log.debug('Starting consumer task')
        task = asyncio.create_task(consume())
        try:
            yield produce
        finally:
            log.debug('Stopping consumer task')
            stop_event.set()
            try:
                async with asyncio.timeout(trailing_timeout):
                    await task
            except TimeoutError as e:
                task.cancel()
                if errors == 'throw':
                    e.add_note("While waiting for trailing messages")
                    raise e
                else:
                    log.warning("Timed out waiting for trailing messages")
