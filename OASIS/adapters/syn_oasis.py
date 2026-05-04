# -*- coding: utf-8 -*-
"""ANTs SyN — delegates to IXI adapter via package import."""
from __future__ import annotations

import os
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from IXI.adapters.syn import build_model, forward  # noqa: E402, F401
