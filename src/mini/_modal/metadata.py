from typing import Callable

import modal
from modal._utils.async_utils import synchronize_api

from mini._modal.model import AppInfo


def _get_metadata(app: modal.app._App) -> AppInfo:
    if not app.app_id or not app._running_app or not app._running_app.app_page_url:
        raise RuntimeError('requires a running app context')
    return AppInfo(
        id=app.app_id,
        name=app.name or '',
        url=app._running_app.app_page_url,
    )


get_metadata: Callable[[modal.app.App], AppInfo] = synchronize_api(_get_metadata)  # type: ignore[assignment]
