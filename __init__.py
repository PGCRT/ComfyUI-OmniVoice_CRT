import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py.omnivoice_nodes import (
    OmniVoiceGenerate,
    OmniVoiceLoadModel,
)

NODE_CLASS_MAPPINGS = {
    "OmniVoiceLoadModel": OmniVoiceLoadModel,
    "OmniVoiceGenerate": OmniVoiceGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceLoadModel": "OmniVoice Load Model",
    "OmniVoiceGenerate": "OmniVoice Generate Audio",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

