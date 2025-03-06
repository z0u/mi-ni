def show(ob, handle: str):
    from IPython.display import display, update_display

    if not hasattr(show, 'handles'):
        show.handles = set()
    if handle in show.handles:
        update_display(ob, display_id=handle)
    else:
        display(ob, display_id=handle)
        show.handles.add(handle)


def displayer():
    handle: str | None = None

    def show(ob):
        nonlocal handle
        from uuid import uuid4 as uuid
        from IPython.display import display, update_display

        if handle is None:
            handle = f'displayer-{uuid().hex[:12]}'
            display(ob, display_id=handle)
        else:
            update_display(ob, display_id=handle)

    return show
