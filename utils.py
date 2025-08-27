
import os
import subprocess
import traceback
import requests
import json
import uuid

from folder_paths import base_path, get_user_directory
from servicestack import JsonServiceClient, WebServiceException, ResponseStatus, EmptyResponse, printdump, from_json

VERSION = 1
DEVICE_ID = None
g_config = {}
g_headers_json={"Content-Type": "application/json"}
g_headers={}

g_logger_prefix = "[agent]"
g_node_dir = os.path.join(get_user_directory(), "comfy_agent")

def urljoin(*args):
    trailing_slash = '/' if args[-1].endswith('/') else ''
    return "/".join([str(x).strip("/") for x in args]) + trailing_slash

def to_error_status(e: Exception, error_code=None, message=None):
    if e is None:
        return None
    if error_code is None:
        error_code = type(e).__name__ or "Exception"
    if message is None:
        message = f"{e}"
    elif message.endswith(":"):
        message += f" {e}"

    if isinstance(e, subprocess.CalledProcessError):
        return ResponseStatus(
            error_code=error_code,
            message=message,
            stack_trace=e.stderr.decode('utf-8') if e.stderr is not None else traceback.format_exc())
    return ResponseStatus(
        error_code,
        message=message,
        stack_trace=traceback.format_exc())

def is_enabled():
    global g_config
    return 'enabled' in g_config and g_config['enabled'] or False

def allow_installing_models():
    global g_config
    return 'install_models' in g_config and g_config['install_models'] or False

def allow_installing_nodes():
    global g_config
    return 'install_nodes' in g_config and g_config['install_nodes'] or False

def allow_installing_packages():
    global g_config
    return 'install_packages' in g_config and g_config['install_packages'] or False

def config_str(name:str):
    return name in g_config and g_config[name] or ""

def get_comfyui_version():
    # comfyui_version.py is a generated file that's sometimes not available
    if os.path.exists(os.path.join(base_path, "comfyui_version.py")):
        from comfyui_version import __version__
        return __version__
    return "unknown"

def create_client():
    client = JsonServiceClient(config_str('url'))
    client.bearer_token = config_str('apikey')
    return client

def headers_json():
    return g_headers_json

def device_id():
    return DEVICE_ID

def _log(message):
    """Helper method for logging from the global polling task."""
    print(f"{g_logger_prefix} {message}")

def _log_error(message, e):
    """Helper method for logging errors from the global polling task."""
    status = None
    if isinstance(e, requests.exceptions.HTTPError):
        try:
            dto = from_json(EmptyResponse, e.response.text)
            status = dto.response_status
        except:
            status = ResponseStatus(error_code='HTTPError', message=e.response.text)
    if isinstance(e, WebServiceException):
        status = e.response_status

    if status is not None:
        error_code = f"[{status.error_code}] " if status.error_code != 'Exception' else ""
        print(f"{g_logger_prefix} {message} {error_code}{status.message}")
    else:
        print(f"{g_logger_prefix} {message} {type(e)} {e}")

def load_config(agent=None, default_config=None):
    global DEVICE_ID, g_config, g_logger_prefix
    g_config = default_config or {'enabled': False}

    if agent is not None:
        g_logger_prefix = f"[{agent}]"
        g_headers["User-Agent"] = g_headers_json["User-Agent"] = f"{agent}/{get_comfyui_version()}/{VERSION}/{DEVICE_ID}"

    try:
        os.makedirs(g_node_dir, exist_ok=True)

        # Read device ID from users/device-id
        device_id_path = os.path.join(g_node_dir, "device-id")
        # check if file exists
        if os.path.isfile(device_id_path):
            with open(device_id_path) as f:
                DEVICE_ID = f.read().strip()
            _log(f"DEVICE_ID: {DEVICE_ID}")
        else:
            # write device id
            _log(f"Generating Device ID at {device_id_path}")
            DEVICE_ID = uuid.uuid4().hex
            with open(device_id_path, "w") as f:
                f.write(DEVICE_ID)
            return
    except IOError:
        _log(f"Failed to read device ID from {device_id_path}")

    try:
        _log("Loading config...")
        config_path = os.path.join(g_node_dir, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path, "r") as f:
            g_config = json.load(f)
    except Exception as e:
        _log(f"Error loading config: {e}")

def save_config(config):
    global g_config
    g_config.update(config)
    os.makedirs(g_node_dir, exist_ok=True)
    _log("Saving config...")
    # _log("Saving config: " + json.dumps(g_config))
    with open(os.path.join(g_node_dir, "config.json"), "w") as f:
        json.dump(g_config, f, indent=4)
