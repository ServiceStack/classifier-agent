from enum import Enum
import io
import os
import threading
import time
import requests
import json
import subprocess
import logging
import traceback
import urllib.parse
import argparse

from servicestack import ResponseStatus

from PIL import Image

from classifier import load_image_models, classify_image
from audio_classifier import convert_to_wav_data, get_audio_tags_from_wav, load_audio_model
from imagehash import phash, dominant_color_hex
from utils import _log_error, create_client, config_str, _log, device_id, headers_json, load_config, to_error_status, paths, Paths

# from server import PromptServer
# from folder_paths import models_dir
# models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")

from dtos import (
    AgentData,
    AssetType,
    GetArtifactClassificationTasks,
    CompleteArtifactClassificationTask,
    Ratings,
)

class AudioBehavior(int, Enum):
    INLINE = 0
    HTTP = 1
    UV = 2
AUDIO_BEHAVIOR = AudioBehavior.INLINE

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
            g_models = load_image_models(models_dir=paths().models, debug=True)
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

                            if AUDIO_BEHAVIOR == AudioBehavior.HTTP:
                                start_time = time.time()
                                response = requests.get('http://localhost:5005/audio/tags', params={"url":url}, headers=headers_json())
                                response.raise_for_status()
                                update.tags = response.json()
                            elif AUDIO_BEHAVIOR == AudioBehavior.UV:
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
                                        g_audio_model = load_audio_model(models_dir=paths().models)
                                        _log(f"Loaded audio model ({'OK' if g_audio_model is not None else 'FAIL'}) in {time.time() - start_time:.2f}s")
                                    except Exception as ex:
                                        _log(f"Error loading audio model: {ex}")

                                if g_audio_model is not None:
                                    try:
                                        start_time = time.time()
                                        file_ext = os.path.splitext(task.url)[1].lower()
                                        format = file_ext[1:]
                                        _log(f"Converting audio to wav: {task.url}")
                                        sample_rate, wav_data = convert_to_wav_data(io.BytesIO(response.content), format=format)
                                        tags = get_audio_tags_from_wav(g_audio_model, sample_rate, wav_data, debug=True)  # noqa: F821
                                        update.tags = tags
                                        _log(f"Classified {type} Artifact {task.id} with {len(update.tags)} tags in {time.time() - start_time:.2f}s")
                                    except Exception as ex:
                                        _log(f"Error getting audio tags: {ex}")
                                else:
                                    _log("No audio model loaded")

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

def setup(use_paths=None):
    global g_client, g_running, g_installed_pip_packages, g_installed_custom_nodes, g_installed_models

    if g_running:
        _log("Already running")
        return

    load_config(agent="classifier-agent", use_paths=use_paths)

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

        global g_categories
        response = g_client.get(AgentData(device_id=device_id()))
        g_categories = response.categories
        _log(f"Categories: {g_categories}")
        return True
    except Exception:
        logging.error("[ERROR] Could not connect to ComfyGateway.")
        logging.error(traceback.format_exc())
        return False

def start(use_paths=None):
    if setup(use_paths):
        _log("Setting up global polling task")
        t = threading.Thread(target=listen_to_messages_poll, daemon=True)
        t.start()
        return t

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
    models_dir = os.path.join(base_path, "models")
    user_dir = os.path.join(base_path, "user")

    parser = argparse.ArgumentParser(description='Classifier Agent')
    parser.add_argument('base_dir',    help='Path to the ComfyUI base directory')
    parser.add_argument('--models_dir', default=models_dir, help='Path to the ComfyUI models directory')
    parser.add_argument('--user_dir',   default=user_dir,   help='Path to the ComfyUI user directory')
    args = parser.parse_args()
    setup(use_paths=Paths(
        base=args.base_dir,
        models=args.models_dir,
        user=args.user_dir
    ))
    listen_to_messages_poll()
