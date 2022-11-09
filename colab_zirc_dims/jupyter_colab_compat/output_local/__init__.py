# -*- coding: utf-8 -*-
"""
Hacky, minimal reproduction of the google.colab package .output module for
local IPython >= 7.0 execution of GUI using the same calls as from the
.output module (e.g.,  redirect_to_element, register_callback, etc.)
"""

from .output_local import *
