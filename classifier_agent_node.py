import io
import os
import threading
import time
import requests
import json
import server # ComfyUI's server instance
import nodes
import subprocess
import logging
import traceback
import base64
import datetime
import urllib.parse

from servicestack import ResponseStatus

from PIL import Image

from .classifier import load_image_models, classify_image
from .audio_classifier import convert_to_wav_data, get_audio_tags_from_wav, load_audio_model
from .imagehash import phash, dominant_color_hex
from .utils import _log_error, create_client, config_str, _log, device_id, headers_json, load_config, to_error_status

# from server import PromptServer
from folder_paths import models_dir

from .dtos import (
    AgentData,
    AssetType,
    GetArtifactClassificationTasks,
    CompleteArtifactClassificationTask,
    Ratings,
)

USE_UV=True

g_client = None
g_models = None
g_audio_model = None
g_categories = []
g_running = False

def is_enabled():
    global g_config
    return True #should have different options to enable classifier vs comfy-agent
    # return 'enabled' in g_config and g_config['enabled'] or False

def listen_to_messages_poll():
    global g_client, g_running
    g_running = True
    g_client = create_client()
    retry_secs = 5
    time.sleep(1)

    global g_models
    if g_models is None:
        try:
            g_models = load_image_models(models_dir=models_dir, debug=True)
        except Exception as ex:
            _log(f"Error loading image models: {ex}")
            g_running = False
            return

    while is_enabled():
        try:
            g_running = True

            _log("Polling for classification tasks")
            request = GetArtifactClassificationTasks(device_id=device_id(),
                types=[AssetType.IMAGE, AssetType.AUDIO])

            response = g_client.get(request)
            retry_secs = 5

            artifact_ids = [f"{task.id}" for task in response.results]

            if response.results is not None and len(response.results) > 0:
                _log(f"Classifying {len(response.results)} artifacts: {','.join(artifact_ids)}")

                for task in response.results:
                    update = CompleteArtifactClassificationTask(artifact_id=task.id, device_id=device_id())
                    type = task.type.value
                    try:
                        _log(f"Classifying {type} {task.url}...")
                        # printdump(task)

                        # download image from url and store in tmp output folder
                        url = urllib.parse.urljoin(config_str('url'), task.url)
                        _log(f"Downloading {url}")
                        if task.type == AssetType.IMAGE:
                            response = requests.get(url, headers=headers_json())
                            response.raise_for_status()

                            with Image.open(io.BytesIO(response.content)) as img:
                                update.phash = f"{phash(img)}"
                                update.color = dominant_color_hex(img)

                                metadata = classify_image(g_models, g_categories, img, debug=True)
                                update.categories = metadata["categories"]
                                update.tags = metadata["tags"]
                                update.objects = metadata["objects"]
                                ratings = metadata["ratings"]
                                update.ratings = Ratings(
                                    predicted_rating=ratings["predicted_rating"],
                                    confidence=ratings["confidence"],
                                    all_scores=ratings["all_scores"])

                            _log(f"Classified {type} Artifact {task.id} with {len(update.tags)} tags, {len(update.objects)} objects, {len(update.categories)} categories")
                        elif task.type == AssetType.AUDIO:

                            if USE_UV:
                                    # Need to run in process because of mediapipe outdated dependency on numpy v1
                                    try:
                                        start_time = time.time()
                                        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "audio_classifier.py")
                                        tags_json = subprocess.check_output(['uv', 'run', script_path, url], text=True)
                                        print(tags_json)
                                        tags = json.loads(tags_json)
                                        update.tags = tags
                                        _log(f"Classified {type} Artifact {task.id} with {len(update.tags)} tags in {time.time() - start_time:.2f}s")
                                    except subprocess.CalledProcessError as e:
                                        _log(f"Error getting audio tags: {e.stderr}")
                                        print("stdout: {e.stdout}")
                                    except Exception as ex:
                                        _log(f"Error getting audio tags: {ex}")
                            else:
                                response = requests.get(url, headers=headers_json())
                                response.raise_for_status()

                                # lasy load Audio Model
                                global g_audio_model
                                if g_audio_model is None:
                                    try:
                                        start_time = time.time()
                                        g_audio_model = load_audio_model(models_dir=models_dir)
                                        _log(f"Loaded audio model in {time.time() - start_time:.2f}s")
                                    except Exception as ex:
                                        _log(f"Error loading audio model: {ex}")

                                if g_audio_model is not None:
                                    try:
                                        start_time = time.time()
                                        file_ext = os.path.splitext(task.url)[1].lower()
                                        format = file_ext[1:]
                                        sample_rate, wav_data = convert_to_wav_data(io.BytesIO(response.content), format=format)
                                        tags = get_audio_tags_from_wav(g_audio_model, sample_rate, wav_data, debug=True)  # noqa: F821
                                        update.tags = tags
                                        _log(f"Classified {type} Artifact {task.id} with {len(update.tags)} tags in {time.time() - start_time:.2f}s")
                                    except Exception as ex:
                                        _log(f"Error getting audio tags: {ex}")

                        else:
                            update.error = ResponseStatus(errorCode="NotImplemented", message=f"Unsupported artifact type: {task.type}")

                    except Exception as ex:
                        _log_error(f"Error classifying {task.id} {task.url}:", ex)
                        logging.error(traceback.format_exc())
                        update.error = to_error_status(ex)
                    finally:
                        # printdump(update)
                        try:
                            g_client.post(update)
                        except Exception as ex:
                            _log_error(f"Error updating Artifact {task.id}:", ex)
            else:
                # sleep 2s
                time.sleep(2)

        except Exception as ex:
            _log(f"Error connecting to {config_str('url')}: {ex}, retrying in {retry_secs}s")
            logging.error(traceback.format_exc())
            time.sleep(retry_secs)  # Wait before retrying
            retry_secs += 5 # Exponential backoff
            g_client = create_client() # Create new client to force reconnect

def start():
    global g_client, g_running, g_installed_pip_packages, g_installed_custom_nodes, g_installed_models

    if g_running:
        _log("Already running")
        return

    load_config(agent="classifier-agent")

    if not device_id():
        return

    if not config_str("url"):
        _log("No URL configured. Please configure in the ComfyAgentNode.")
        return
    if not config_str("apikey"):
        _log("No API key configured. Please configure in the ComfyAgentNode.")
        return
    if not is_enabled():
        _log("Autostart is disabled. Enable in the ComfyAgentNode.")
        return

    g_running = True
    g_client = create_client()

    try:
        try:

            global g_categories
            response = g_client.get(AgentData(device_id=device_id()))
            g_categories = response.categories
            _log(f"Categories: {g_categories}")

            _log("Setting up global polling task.")
            # register_agent()
            # listen to messages in a background thread
            t = threading.Thread(target=listen_to_messages_poll, daemon=True)
            t.start()

        except Exception:
            logging.error("[ERROR] Could not connect to ComfyGateway.")
            logging.error(traceback.format_exc())

    except Exception:
        logging.error("[ERROR] Could not load models.")
        logging.error(traceback.format_exc())


# --- Autostart Logic ---
start()

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
