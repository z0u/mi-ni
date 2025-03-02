import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncGenerator, Callable, Literal, TypeAlias, TypeVar

import modal

T = TypeVar('T')
Handler: TypeAlias = Callable[[list[T]], None]

log = logging.getLogger(__name__)

# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
MAX_LEN = 5_000


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
            # separate control partition of that queue. Using a control channel
            # avoids the need for polling and timeouts.

            # Emit values.
            # TODO: Allow caller to provide many values at once.
            q.put_many([value])

            # Notify consumer.
            q.put(True, partition='signal')

        async def consume() -> None:
            """Take values from the queue until the context manager exits"""
            # This function is not exposed, so there's exactly one consumer.
            # It always runs locally.

            while True:
                # Wait until values are produced or the context manager exits.
                get_task = asyncio.create_task(
                    q.get_many.aio(MAX_LEN, partition='signal'))
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    [get_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Either way, get all available messages.
                values = await q.get_many.aio(MAX_LEN, block=False)
                for value in values:
                    # TODO: Allow receiver to consume many values at once.
                    receive(value)

                # Can't just check stop_event.is_set here; we need to know
                # whether it was set before the last batch.
                if stop_task in done:
                    await q.clear.aio(all=True)
                    break

        log.debug('Starting consumer task')
        task = asyncio.create_task(consume())
        try:
            yield produce
        finally:
            log.debug('Stopping consumer task')
            stop_event.set()
            try:
                await asyncio.wait_for(task, trailing_timeout)
            except TimeoutError as e:
                if errors == 'throw':
                    e.add_note("While waiting for trailing messages")
                    raise e
                else:
                    log.warning("Timed out waiting for trailing messages")
