#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证模块
"""

import os
import sys

# Add project root directory to system path
_exec_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_exec_path)

from module.utils.log import get_logger
logger = get_logger(__name__)


def val() -> bool:
    return True