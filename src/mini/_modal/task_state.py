import asyncio
import logging
import time
from html import escape as html_escape
from typing import Sequence
from urllib.parse import urlparse

from mini._modal.model import AppInfo, State, StateUpdate, TaskInfo
from utils.nb import displayer

log = logging.getLogger(__name__)


UL_STYLE = """display: inline-block; list-style-type: none; padding: 0; margin: 0;"""
LI_STYLE = """display: inline-block; margin-right: 10px; font-size: 1.2em;"""


def app_state_vis(app_info: AppInfo, rate_limit: float = 1.0):
    show = displayer()

    url = urlparse(app_info.url)
    if url.scheme not in ('http', 'https'):
        raise ValueError(f'Invalid URL scheme. Expected "http" or "https". {app_info.url}')

    from IPython.display import HTML

    _tasks: list[TaskInfo] = []
    _message = ''

    last_render_time = time.monotonic() - rate_limit
    update_task: asyncio.Task | None = None
    lock = asyncio.Lock()

    def _render_html():
        task_list = []
        for task in _tasks:
            title, icon = describe_task(task)
            task_list.append(f'<li title="{html_escape(title)}" style="{html_escape(LI_STYLE)}">{icon}</li>')

        html = f"""
        <a href="{html_escape(str(url))}" title="View Modal dashboard for this app">Running {html_escape(app_info.name)}</a>.
        Tasks:
        <ul style="{html_escape(UL_STYLE)}">
            {''.join(task_list) if task_list else f'<li title="No tasks" style="{html_escape(LI_STYLE)}">...</li>'}
        </ul>
        <p>{html_escape(_message) if _message else '&nbsp;'}</p>
        """
        show(HTML(html))

    async def update(tasks: Sequence[TaskInfo] | None = None, message: str | None = None):
        nonlocal _message, _tasks, last_render_time, update_task

        async with lock:
            if tasks is not None:
                _tasks = list(tasks)
            if message is not None:
                _message = message.strip()

            if update_task and not update_task.done():
                # An update is already scheduled. The new data will be picked up.
                return

            now = time.monotonic()
            if now - last_render_time >= rate_limit:
                # More than 1s since last update, so update now.
                last_render_time = now
                _render_html()
            else:
                # Schedule an update
                delay = rate_limit - (now - last_render_time)
                update_task = asyncio.create_task(_delayed_update(delay))

    async def _delayed_update(delay: float):
        nonlocal last_render_time, update_task
        await asyncio.sleep(delay)
        async with lock:
            last_render_time = time.monotonic()
            _render_html()
            update_task = None

    return update


def describe_task(task: TaskInfo) -> tuple[str, str]:
    if task.state == State.PENDING:
        title = 'Pending'
        icon = '🔜'
    elif task.state == State.CREATING:
        if task.root_function_id:
            # User defined function, use gear icon.
            title = 'Creating task for function'
            icon = '⚙️'
        else:
            # A task with no function id is probably building an image. Use hammer icon.
            title = 'Building image'
            icon = '🔨'
    elif task.state == State.ACTIVE:
        title = 'Running'
        icon = '▶️'
    elif task.state == State.COMPLETED:
        title = 'Completed'
        icon = '✅'
    else:
        title = 'Unknown state'
        icon = '❓'
    return title, icon


class TaskStateTracker:
    """
    Tracks the lifecycle and state of multiple Modal tasks.

    Instantiate this class and pass `LogsItem` objects to its `update()`
    method to keep track of the state of each running task.
    """

    # Mapping from the TaskState enum integer to a readable string.

    def __init__(self):
        """Initializes the tracker with an empty dictionary to store tasks."""
        self._tasks: dict[str, TaskInfo] = {}

    @property
    def tasks(self) -> Sequence[TaskInfo]:
        return list(self._tasks.values())

    def update(self, item: StateUpdate):
        """Process a StateChange to update the state of a task."""
        task_id = item.task_id
        if not task_id:
            # Some log items are not tied to a specific task. We can ignore them for state tracking.
            return

        if task_id in self._tasks:
            task_info = self._tasks[task_id]
            if task_info.root_function_id != item.root_function_id:
                log.warning(
                    f'Task {task_id} function ID changed from {task_info.root_function_id} to {item.root_function_id}'
                )
                task_info.root_function_id = item.root_function_id
            if task_info.state != item.state:
                log.info(f'Task {task_id} state changed from {task_info.state} to {item.state}')
                task_info.state = item.state
                task_info.last_update = item.timestamp
        else:
            log.info(f'Task {task_id} initialized as {item.state}')
            self._tasks[task_id] = TaskInfo(
                task_id=task_id,
                root_function_id=item.root_function_id,
                state=item.state,
                last_update=item.timestamp,
            )
