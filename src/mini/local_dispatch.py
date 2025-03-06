import asyncio
from functools import wraps
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, TypeVar

import modal

from mini.types import Q_MAX_LEN, AsyncHandler, Handler, SyncHandler
from mini.utils import coerce_to_async

log = logging.getLogger(__name__)

T = TypeVar('T')


@asynccontextmanager
async def send_batch_to(
    receive: Handler[list[T]],
    trailing_timeout: float | None = 5,
) -> AsyncGenerator[SyncHandler[list[T]]]:
    """
    Create a distributed producer-consumer for batch processing with Modal.

    This async context manager sets up a distributed queue system where multiple
    producers can send batches of values to a single consumer function. The context
    yields a function that producers can call to send batches of values.

    Inside the context, a consumer task continuously reads batches from the queue
    and processes them using the provided `receive` function. The consumer will
    continue processing values until the context is exited and any trailing values
    are handled.

    Args:
        receive: A function that processes batches of values. Will be called
            with each batch of values as they become available. Can be either
            synchronous or asynchronous — it will be called appropriately based
            on its type.
        trailing_timeout: Number of seconds to wait for trailing messages after
            the context manager exits. If None, waits indefinitely.
        errors: How to handle errors in trailing message processing:
            - 'throw': Raises a TimeoutError if trailing message processing times out
            - 'log': Logs a warning if trailing message processing times out

    Yields:
        send: A function that accepts a list of values to send to the consumer.
            This function can be called from multiple distributed workers.

    Example:
        ```python
        async def process_batch(items: list[str]) -> None:
            print(f"Processing {len(items)} items")

        async with _send_batch_to(process_batch) as send_batch:
            # This can be called from multiple distributed workers
            send_batch(["item1", "item2", "item3"])
        ```

    """
    async with modal.Queue.ephemeral() as q:
        # Wrap, but remove the reference to the wrapped function so it doesn't get serialized.
        produce = wraps(receive)(_producer_batch(q))
        del produce.__wrapped__

        consume, stop = _batched_consumer(q, receive)

        log.debug('Starting consumer task')
        task = asyncio.create_task(consume())
        try:
            # The caller can send this to remote workers to put messages on the queue.
            yield produce
        finally:
            log.debug('Stopping consumer task')
            stop()
            try:
                await asyncio.wait_for(task, trailing_timeout)
            except TimeoutError:
                log.warning('Timed out waiting for trailing messages')


def _producer_batch(q: modal.Queue):
    def produce_batch(values: list[T]) -> None:
        """Send values to the consumer."""
        # This function is yielded as the context, so there may be several
        # distributed producers. It gets pickled and sent to remote workers
        # for execution, so we can't use local synchronization mechanisms.
        # All we have is a distributed queue - but we can send signals on a
        # separate control partition of that queue. Using a control channel
        # avoids the need for polling and timeouts.

        # Emit values.
        q.put_many(values)

        # Notify consumer.
        q.put(True, partition='signal')

    return produce_batch


def _batched_consumer(q: modal.Queue, receive: Handler[list[T]]):
    areceive: AsyncHandler[list[T]] = coerce_to_async(receive)
    stop_event = asyncio.Event()

    async def batched_consume() -> None:
        """Take values from the queue until the context manager exits."""
        # This function is not exposed, so there's exactly one consumer.
        # It always runs locally.

        while True:
            # Wait until values are produced or the context manager exits.
            get_task = asyncio.create_task(q.get_many.aio(Q_MAX_LEN, partition='signal'))
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                [get_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Either way, get all available messages.
            values: list[T] = await q.get_many.aio(Q_MAX_LEN, block=False)
            if values:
                await areceive(values)

            # Can't just check stop_event.is_set here; we need to know
            # whether it was set before the last batch.
            if stop_task in done:
                await q.clear.aio(all=True)
                break

    def stop():
        """Stop the consumer."""
        stop_event.set()

    return batched_consume, stop
