
import os
import re
import subprocess
import traceback
import requests
import json
import uuid
import sys
import shutil

# from folder_paths import base_path, get_user_directory
from servicestack import JsonServiceClient, WebServiceException, ResponseStatus, EmptyResponse, from_json

VERSION = 1
DEVICE_ID = None

class Paths(object):
    def __init__(self, base, models=None, user=None, agent=None):
        self.base = base
        self.models = models or os.path.join(base, "models")
        self.user = user or os.path.join(base, "user")
        self.agent = agent or os.path.join(self.user, "comfy_agent")

g_config = {}
g_headers_json={"Content-Type": "application/json"}
g_headers={}

g_logger_prefix = "[agent]"
g_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
g_paths = Paths(g_base_path)

def paths():
    global g_paths
    return g_paths

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
    if os.path.exists(os.path.join(g_base_path, "comfyui_version.py")):
        # from comfyui_version import __version__
        # return __version__
        # use regex to extract version
        with open(os.path.join(g_base_path, "comfyui_version.py")) as f:
            version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)
            return version
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
    print(f"{g_logger_prefix} {message}", flush=True)

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
        print(f"{g_logger_prefix} {message} {error_code}{status.message}", flush=True)
    else:
        print(f"{g_logger_prefix} {message} {type(e)} {e}", flush=True)

def load_config(agent=None, default_config=None, use_paths=None):
    global DEVICE_ID, g_config, g_logger_prefix, g_base_path, g_paths
    g_config = default_config or {'enabled': False}

    if use_paths is not None:
        if isinstance(use_paths, Paths):
            g_paths = use_paths
        else:
            g_paths = Paths(use_paths)

    if agent is not None:
        g_logger_prefix = f"[{agent}]"
        g_headers["User-Agent"] = g_headers_json["User-Agent"] = f"{agent}/{get_comfyui_version()}/{VERSION}/{DEVICE_ID}"

    try:
        os.makedirs(g_paths.agent, exist_ok=True)

        # Read device ID from users/device-id
        device_id_path = os.path.join(g_paths.agent, "device-id")
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
        config_path = os.path.join(g_paths.agent, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path, "r") as f:
            g_config = json.load(f)
    except Exception as e:
        _log(f"Error loading config: {e}")

def save_config(config):
    global g_config
    g_config.update(config)
    os.makedirs(g_paths.agent, exist_ok=True)
    _log("Saving config...")
    # _log("Saving config: " + json.dumps(g_config))
    with open(os.path.join(g_paths.agent, "config.json"), "w") as f:
        json.dump(g_config, f, indent=4)

def detect_python_environment():
    """
    Detect the current Python environment and return appropriate pip command.
    Returns a list of command components for subprocess.run()
    """
    python_exe = sys.executable

    # Check for conda environment
    if 'CONDA_DEFAULT_ENV' in os.environ or 'CONDA_PREFIX' in os.environ:
        # Try to find conda executable
        conda_exe = shutil.which('conda')
        if conda_exe:
            _log(f"Detected conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
            return [conda_exe, 'run', '-n', os.environ.get('CONDA_DEFAULT_ENV', 'base'), 'pip']
        else:
            _log("Conda environment detected but conda executable not found, falling back to python -m pip")

    # Check for uv environment
    if 'UV_PROJECT_ROOT' in os.environ or shutil.which('uv'):
        uv_exe = shutil.which('uv')
        if uv_exe:
            _log("Detected uv environment")
            return [uv_exe, 'pip']
        else:
            _log("UV environment detected but uv executable not found, falling back to python -m pip")

    # Check for virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        _log(f"Detected virtual environment: {sys.prefix}")

    # Default to using the current Python executable with pip module
    _log(f"Using Python executable: {python_exe}")
    return [python_exe, '-m', 'pip']

def get_pip_install_command(package_or_requirements_file, is_requirements_file=False):
    """
    Get the appropriate pip install command for the current environment.

    Args:
        package_or_requirements_file: Package name or path to requirements.txt
        is_requirements_file: Whether the input is a requirements file

    Returns:
        List of command components for subprocess.run()
    """
    pip_cmd = detect_python_environment()

    if is_requirements_file:
        return pip_cmd + ['install', '-r', package_or_requirements_file]
    else:
        return pip_cmd + ['install', package_or_requirements_file]

def get_pip_uninstall_command(package_name):
    """
    Get the appropriate pip uninstall command for the current environment.

    Args:
        package_name: Name of the package to uninstall

    Returns:
        List of command components for subprocess.run()
    """
    pip_cmd = detect_python_environment()
    return pip_cmd + ['uninstall', '-y', package_name]

def utf8str(strOrBytes):
    if strOrBytes is None:
        return ''
    s = ''
    if isinstance(strOrBytes, str):
        s = strOrBytes
    else:
        s = strOrBytes.decode('utf-8')
    if isinstance(s, str):
        s = s.strip()
    return s
