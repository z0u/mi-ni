"""Test configuration and compatibility shims."""

import sys
import typing

if sys.version_info >= (3, 14):
    # pydantic ≤2.13 calls typing._eval_type(prefer_fwd_module=True) for Python 3.14+,
    # but 3.14rc2 renamed that parameter to parent_fwdref. Patch it here until pydantic
    # ships a compatible release.
    _real_eval_type = typing._eval_type  # type: ignore

    def _eval_type_compat(t, globalns, localns, type_params=None, *, prefer_fwd_module=None, **kw):
        if prefer_fwd_module is not None and 'parent_fwdref' not in kw:
            kw['parent_fwdref'] = prefer_fwd_module
        return _real_eval_type(t, globalns, localns, type_params=type_params, **kw)

    typing._eval_type = _eval_type_compat  # type: ignore
