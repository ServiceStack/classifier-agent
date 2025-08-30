import os
import subprocess

from folder_paths import base_path, models_dir, get_user_directory
from .utils import Paths, _log, load_config

# --- Autostart Logic ---
def start(use_paths=None):
    cmd = [
        'uv',
        'run',
        os.path.join(os.path.dirname(__file__), 'classifier_agent.py'),
        use_paths.base,
        '--models_dir', use_paths.models,
        '--user_dir',   use_paths.user,
    ]
    # run subprocess, piping stdout and stderr to os
    load_config(agent="classifier-agent", use_paths=use_paths)
    _log(f"Starting classifier agent with command: {' '.join(cmd)}")
    subprocess.Popen(cmd)

start(use_paths=Paths(base = base_path,
                      models = models_dir,
                      user = get_user_directory()))

class ClassifiyImage:
    NODE_NAME = "ClassifiyImage"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "classify"
    CATEGORY = "comfy_agent"

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    def classify(self, url, node_id, version=""):
        _log(f"Classifying {url}")
        return ()

class ClassifiyAudio:
    NODE_NAME = "ClassifiyAudio"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "classify"
    CATEGORY = "comfy_agent"

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    def classify(self, url, node_id, version=""):
        _log(f"Classifying {url}")
        return ()

# --- ComfyUI Registration ---
class RegisterClassifierNodes:
    NODE_CLASS_MAPPINGS = {
        ClassifiyImage.NODE_NAME: ClassifiyImage,
        ClassifiyAudio.NODE_NAME: ClassifiyAudio,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        ClassifiyImage.NODE_NAME: "Classifiy Image",
        ClassifiyAudio.NODE_NAME: "Classifiy Audio",
    }
