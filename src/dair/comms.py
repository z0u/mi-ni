import asyncio
from contextlib import asynccontextmanager
import inspect
import logging
from typing import AsyncGenerator, Literal, Protocol, TypeVar

import modal

T = TypeVar('T')


class SyncHandler(Protocol[T]):
    def __call__(self, values: T) -> None: ...


class AsyncHandler(Protocol[T]):
    async def __call__(self, values: T) -> None: ...


log = logging.getLogger(__name__)

# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
MAX_LEN = 5_000


@asynccontextmanager
async def send_batch_to(
    receive: SyncHandler[list[T]] | AsyncHandler[list[T]],
    trailing_timeout: float | None = 5,
    errors: Literal['throw', 'log'] = 'log'
) -> AsyncGenerator[SyncHandler[list[T]]]:
    """
    Create a distributed producer-consumer pattern for batch processing with Modal.

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
    # There can be many producers, but only one consumer.

    is_async_handler = inspect.iscoroutinefunction(receive)
    stop_event = asyncio.Event()

    async with modal.Queue.ephemeral() as q:
        def produce(values: list[T]) -> None:
            """Send values to the consumer."""
            # This function is yielded as the context, so there may be several
            # distributed producers. It gets pickled and sent to remote workers
            # for execution, so we can't use local synchronization mechanisms.
            # All we have is a distributed queue - but we can send signals on a
            # separate control partition of that queue. Using a control channel
            # avoids the need for polling and timeouts.

            # Emit values.
            # TODO: Allow caller to provide many values at once.
            q.put_many(values)

            # Notify consumer.
            q.put(True, partition='signal')

        async def consume() -> None:
            """Take values from the queue until the context manager exits."""
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
                values: list[T] = await q.get_many.aio(MAX_LEN, block=False)
                if values:
                    if is_async_handler:
                        await receive(values)
                    else:
                        receive(values)

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


@asynccontextmanager
async def send_to(
    receive: SyncHandler[T] | AsyncHandler,
    trailing_timeout: float | None = 5,
    errors: Literal['throw', 'log'] = 'log'
) -> AsyncGenerator[SyncHandler[T]]:
    """
    Create a distributed producer-consumer pattern for single-item processing with Modal.

    Inside the context, a consumer task continuously reads items from the queue
    and processes them using the provided `receive` function. The consumer will
    continue processing values until the context is exited and any trailing values
    are handled.

    For batch processing, use `send_to.batch` which accepts and processes lists of items.

    Args:
        receive: A function that processes a single value. Will be called
            with each value as it becomes available. Can be either
            synchronous or asynchronous — it will be called appropriately based
            on its type.
        trailing_timeout: Number of seconds to wait for trailing messages after
            the context manager exits. If None, waits indefinitely.
        errors: How to handle errors in trailing message processing:
            - 'throw': Raises a TimeoutError if trailing message processing times out
            - 'log': Logs a warning if trailing message processing times out

    Yields:
        send: A function that accepts a single value to send to the consumer.
            This function can be called from multiple distributed workers.

    Example:
        ```python
        async def process_item(item: str) -> None:
            print(f"Processing {item}")

        async with send_to(process_item) as send:
            # This can be called from multiple distributed workers
            send("item1")

        # For batch processing, use send_to.batch instead
        async with send_to.batch(process_batch) as send_batch:
            send_batch(["item1", "item2", "item3"])
        ```

    """
    async with send_batch_to(receive=receive, trailing_timeout=trailing_timeout, errors=errors) as produce_batch:
        def produce(value: T) -> None:
            """Send a single value to the consumer."""
            produce_batch([value])
        yield produce


send_to.batch = send_batch_to


@asynccontextmanager
async def run(app: modal.App, trailing_timeout=10):
    """
    Run a Modal app and display its stdout stream.

    This differs from `modal.enable_output`, in that this function only shows logs from inside the container.

    Args:
        app: The Modal app to run.
        trailing_timeout: Number of seconds to wait for trailing logs after the app exits.
    """

    async def consume():
        async for output in app._logs.aio():
            if output == "Stopping app - local entrypoint completed.\n":
                # Consume this infrastructure message
                continue
            # Don't add newlines, because the output contains control characters
            print(output, end="")
            # No need to break: the loop should exit when the app is done

    # 1. Start the app
    # 2. Start consuming logs
    # 3. Yield control to the caller
    # 4. Wait for the logs to finish

    async with app.run():
        task = asyncio.create_task(consume())
        yield

    # Can't wait inside the context manager, because the app would still be running
    try:
        await asyncio.wait_for(task, timeout=trailing_timeout)
    except asyncio.TimeoutError as e:
        e.add_note("While waiting for trailing stdout")
        raise e
