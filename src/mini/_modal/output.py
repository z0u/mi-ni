import logging
import sys
from textwrap import indent
from typing import Any, AsyncGenerator, Callable, cast

import modal
from modal._utils.async_utils import synchronize_api
from modal_proto import api_pb2

from mini._modal.model import FD, LogsItem, State, StateUpdate
from utils.logging import TRACE

log = logging.getLogger(__name__)


def basic_output_handler(log_item: LogsItem) -> None:
    """Default log handler that simply prints the log message."""
    if not log_item.data:
        return
    if log_item.fd == FD.STDOUT:
        print(log_item.data, file=sys.stdout, end='')
    elif log_item.fd == FD.STDERR:
        print(log_item.data, file=sys.stderr, end='')
    else:
        # For 'infra' logs, we can choose to log differently or ignore
        log.debug('infra message: %s', log_item.data)


async def _stream_logs(app: modal.app._App) -> AsyncGenerator[StateUpdate | LogsItem, Any]:
    """Stream raw logs from the app."""
    # This is a re-implementation of modal.App._logs to get structured log data.

    # Guaranteed to be available within the context of app.run()
    app_id = app.app_id
    client = app._client
    if not app_id or not client:
        raise RuntimeError('requires a running/stopped app')

    last_log_batch_entry_id: str | None = None
    while True:
        request = api_pb2.AppGetLogsRequest(
            app_id=app_id,
            timeout=55,
            last_entry_id=last_log_batch_entry_id,  # type: ignore[assignment]
        )
        async for log_batch in client.stub.AppGetLogs.unary_stream(request):
            log.log(TRACE, 'log batch:\n%s', log_batch)
            # log.info('log batch:\n%s', indent(str(log_batch), '  '))
            if not isinstance(log_batch, api_pb2.TaskLogsBatch):
                log.warning(
                    'unexpected log batch type: %s. Expected %s.',
                    type(log_batch).__name__,
                    api_pb2.TaskLogsBatch.__name__,
                )
                continue

            if log_batch.entry_id:
                last_log_batch_entry_id = log_batch.entry_id

            if log_batch.app_done:
                return

            for _log in log_batch.items:
                _log = cast(api_pb2.TaskLogs, _log)

                if _log.task_state:
                    yield StateUpdate(
                        task_id=log_batch.task_id,
                        root_function_id=log_batch.root_function_id,
                        state=State.from_proto(_log.task_state),
                        timestamp=_log.timestamp,
                    )

                if _log.data:
                    yield LogsItem(
                        task_id=log_batch.task_id,
                        root_function_id=log_batch.root_function_id,
                        data=_log.data,
                        fd=FD.from_proto(_log.file_descriptor),
                        timestamp=_log.timestamp,
                    )


stream_logs: Callable[[modal.app.App], AsyncGenerator[StateUpdate | LogsItem, Any]] = synchronize_api(_stream_logs).aio  # type: ignore[assignment]


# def patch_app():
#     print('Patching modal.App...')
#     modal.app._App._logs = _logs  # type: ignore[attr-defined]
#     modal.app.App._logs = modal._utils.async_utils.synchronize_api(modal.app._App._logs, modal.app)  # type: ignore[attr-defined]
#     print(
#         'Patching complete.',
#         getattr(modal.app._App, '_logs', None),
#         getattr(modal.App, '_logs', None),
#     )
