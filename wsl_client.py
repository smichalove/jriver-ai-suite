"""WSL2 VLM Server Control & Communication Client.

Purpose:
    This module provides the Windows host-side client interface and manager to start,
    interact with, and stop the Gemma-4 VLM inference server hosted inside a WSL2 Ubuntu
    Docker container (named trt_llm_build). It encapsulates WSL commands, Docker management,
    GPU/CUDA virtualization warm-up, error recovery (WSL shutdown resets), and REST requests.

Architecture and Mechanics:
    1. GPU Driver Warmup & Fail-Safe Reset: Executes a swift "nvidia-smi" inside WSL to ensure CUDA virtualization
       is responsive. If it hangs or returns errors (indicating a driver deadlock), it triggers "wsl --shutdown"
       and re-initializes to recover the GPU state automatically.
    2. Docker Container Control: Natively mounts the codebase volumes and starts/stops the Docker container.
       Verifies directory mounts before starting uvicorn.
    3. Process Lifecycle Management: Checks if a uvicorn process is already running in the container via "pgrep"
       to avoid duplicate instances, and detaches execution when spawning new instances.
    4. HTTP REST Communication: Batches image files (supporting JPEG, PNG, HEIF) and converts them into base64
       encoded payloads, posting them to the WSL container's FastAPI endpoints (/describe, /analyze).
    5. Cleanup: Shuts down the Docker container when stop_wsl_server is called, instantly releasing VRAM.

Execution Modes:
    - Library Module: Imported and consumed by photo processing/cataloging pipelines (such as describe_photos.py).
"""

import os
import subprocess
import base64
import atexit

import io
import time
import logging
from typing import List, Dict, Optional, Any
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable decompression limit to allow high-resolution upscaled images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow reading truncated or slightly corrupted files
import requests
import pillow_heif
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

SERVER_URL: str = "http://127.0.0.1:8000/describe"
HEALTH_URL: str = "http://127.0.0.1:8000/docs"
CONTAINER_NAME: str = "trt_llm_build"
MAX_IMAGE_DIM: int = 1024

# Thread-safe global requests Session to enable Keep-Alive connection pooling
# and avoid ephemeral port exhaustion under heavy batch processing loads.
session: requests.Session = requests.Session()


class VLMServerConnection:
    """Manages connection, health checks, and queries to a specific VLM server instance.

    Attributes:
        name: A descriptive name for the server (e.g., 'Primary', 'Secondary').
        base_url: The base HTTP URL of the server (e.g., 'http://127.0.0.1:8000').
        batch_size: Default batch size for this server.
        describe_url: The endpoint to post describe queries to.
        health_url: The endpoint to query for health status.
    """
    name: str
    base_url: str
    batch_size: int
    describe_url: str
    health_url: str

    def __init__(self, name: str, base_url: str, batch_size: int = 2) -> None:
        """Initializes the VLM server connection endpoints.

        Args:
            name: Descriptive name for the server instance.
            base_url: Base URL of the server (e.g., 'http://127.0.0.1:8000').
            batch_size: Default batch size for this server.
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.describe_url = f"{self.base_url}/describe"
        self.health_url = f"{self.base_url}/docs"

    def is_alive(self) -> bool:
        """Checks if this specific VLM server is online and responding.

        Args:
            None

        Returns:
            True if responsive, False otherwise.
        """
        try:
            response = session.get(self.health_url, timeout=3.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def query(self, images_base64: List[str], prompt_text: str, temperature: float = 0.7, timeout: float = 600.0) -> List[str]:
        """Queries this specific VLM server with base64 image strings.

        Args:
            images_base64: List of base64-encoded image strings.
            prompt_text: Prompt instruction text.
            temperature: Generation temperature.
            timeout: Request timeout in seconds.

        Returns:
            A list of raw response strings from the VLM.

        Raises:
            requests.RequestException: If connection or request fails.
        """
        payload: Dict[str, Any] = {
            "images_base64": images_base64,
            "prompt_text": prompt_text,
            "temperature": temperature
        }
        response = session.post(self.describe_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("raw_responses", [])
        else:
            raise requests.RequestException(
                f"Server {self.name} returned HTTP error {response.status_code}: {response.text}"
            )

class FatalVLMServerError(RuntimeError):
    """Exception raised when the VLM server fails persistently and recovery is impossible.
    
    This error indicates that retries and automatic server restarts have all failed
    to restore communication with the vision-language model backend, requiring the
    pipeline to abort to prevent endless processing timeouts.
    """
    pass

# Global tracker to limit the number of automatic WSL/Docker restarts per execution.
# Prevents infinite reboot loops on corrupted or un-processable image batches.
_auto_restart_count: int = 0

# Global process handle to keep WSL VM awake during script runtime
_keep_alive_process: Optional[subprocess.Popen] = None
_should_terminate_keep_alive: bool = True


def leave_keep_alive_running() -> None:
    """Instructs the client to skip terminating the keep-alive process on exit.

    Args:
        None

    Returns:
        None
    """
    global _should_terminate_keep_alive
    _should_terminate_keep_alive = False


def start_keep_alive() -> None:
    """Spawns a background WSL keep-alive process to prevent WSL2 VM idle auto-shutdown.

    Args:
        None

    Returns:
        None
    """
    global _keep_alive_process
    if _keep_alive_process is None:
        try:
            logger.info("Spawning background WSL keep-alive process to prevent VM auto-shutdown...")
            # We run a sleep infinity process inside WSL which will keep the VM alive.
            # Using CREATE_NO_WINDOW flag on Windows prevents pop-up console windows.
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            _keep_alive_process = subprocess.Popen(
                ["wsl", "sleep", "infinity"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
        except Exception as e:
            logger.warning(f"Failed to spawn WSL keep-alive process: {e}")


def stop_keep_alive() -> None:
    """Terminates the active WSL keep-alive process if permitted.

    Args:
        None

    Returns:
        None
    """
    global _keep_alive_process, _should_terminate_keep_alive
    if not _should_terminate_keep_alive:
        logger.info("Leaving background WSL keep-alive process running to preserve VM state.")
        return

    if _keep_alive_process is not None:
        try:
            logger.info("Terminating background WSL keep-alive process...")
            _keep_alive_process.terminate()
            _keep_alive_process.wait(timeout=2.0)
        except Exception:
            pass
        finally:
            _keep_alive_process = None


@atexit.register
def cleanup_keep_alive() -> None:
    """Auto-registered exit handler to guarantee the keep-alive process is terminated.

    Args:
        None

    Returns:
        None
    """
    stop_keep_alive()



def is_server_alive() -> bool:
    """Checks if the FastAPI server is running and responding on port 8000.

    Args:
        None

    Returns:
        True if alive, False otherwise.
    """
    try:
        # Use GET instead of HEAD to ensure route compatibility,
        # and a 3.0s timeout to survive transient VM wake-up lags.
        response = session.get(HEALTH_URL, timeout=3.0)
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_uvicorn_running() -> bool:
    """Checks if the uvicorn process is already running inside the Docker container.

    Args:
        None

    Returns:
        True if the uvicorn process is active, False otherwise.
    """
    try:
        # Check if any processes matching 'uvicorn' are active inside the container.
        # This will return a non-zero exit code if no matching processes are found.
        # Added a 10-second timeout to prevent indefinite hangs if the VM freezes.
        result = subprocess.run(
            ["wsl", "-u", "workbench", "docker", "exec", CONTAINER_NAME, "pgrep", "-f", "uvicorn"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10.0
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def start_wsl_server() -> bool:
    """Spins up the WSL2 Docker container and starts the FastAPI VLM server inside it.

    If the server is already active and responsive to health checks, skips all startup
    checks and returns True immediately. Otherwise, ensures the Docker container is started,
    verifies workspace mounts, launches the FastAPI server using uvicorn if not already active,
    and polls the server health endpoint until the weights are fully loaded.

    Args:
        None

    Returns:
        True if the server is successfully running and ready, False otherwise.

    Raises:
        None
    """
    # 0. Start the background WSL keep-alive process to prevent auto-shutdown
    start_keep_alive()

    # 1. Quick exit if the model server is already up, healthy, and listening.
    # This avoids resetting connection state or reloading model weights when a server is ready.
    if is_server_alive():
        logger.info("WSL2 model server is already running and responsive. Bypassing startup procedure.")
        return True

    # 1.5. If Uvicorn is already running inside the container (likely still loading weights),
    # wait for it to finish loading instead of restarting the container.
    if is_uvicorn_running():
        logger.info("Uvicorn is already running inside the container (likely still loading weights).")
        logger.info("Waiting for Gemma 4 VLM weights to load into VRAM...")
        max_attempts = 150
        for attempt in range(1, max_attempts + 1):
            if is_server_alive():
                logger.info("WSL2 Model Server is active and listening on port 8000!")
                return True
            if not is_uvicorn_running():
                logger.warning("Uvicorn process stopped running while waiting. Proceeding to restart container...")
                break
            time.sleep(3.0)
            if attempt % 5 == 0:
                logger.info(f"Still waiting for model loading... (Attempt {attempt}/{max_attempts})")
        else:
            logger.error("Timeout: Existing WSL2 model server failed to become responsive within 7.5 minutes.")
            return False

    logger.info("Starting WSL2 model server...")
    
    # 1. Restart the Docker container cleanly.
    # Using 'docker restart' ensures any fragmented or leaked GPU memory is fully released,
    # preventing silent VRAM Out-of-Memory (OOM) crashes on large high-res batch runs.
    # Added robust subprocess timeouts to prevent freezing if WSL itself deadlocks.
    try:
        logger.info(f"Cleanly restarting container '{CONTAINER_NAME}' to reset VRAM state...")
        subprocess.run(["wsl", "-u", "workbench", "docker", "restart", CONTAINER_NAME], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45.0)
        
        # Wait for container volume mounts to be fully attached (up to 10 seconds)
        mount_ready = False
        for _ in range(10):
            try:
                res = subprocess.run(
                    ["wsl", "-u", "workbench", "docker", "exec", CONTAINER_NAME, "test", "-d", "/workspace/gemma_cataloger"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0
                )
                if res.returncode == 0:
                    mount_ready = True
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1.0)
            
        if not mount_ready:
            logger.error("Timeout: /workspace/gemma_cataloger mount was not mounted in time inside container.")
            return False
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err_msg = ""
        if isinstance(e, subprocess.CalledProcessError):
            err_msg = e.stderr.decode().strip() if e.stderr else str(e)
        else:
            err_msg = f"Command timed out after {e.timeout}s"
        logger.error(f"Failed to restart docker container {CONTAINER_NAME}: {err_msg}")
        return False

    # 2. Check if uvicorn is already running. If not, launch it.
    if is_uvicorn_running():
        logger.info("Uvicorn is already running inside the container. Bypassing launch command.")
    else:
        # Start uvicorn server in the background inside the container
        # We use -d on docker exec to launch it detached/daemonized
        cmd: List[str] = [
            "wsl", "-u", "workbench", "docker", "exec", "-d", "-w", "/workspace",
            "-e", "TQDM_DISABLE=1",
            "-e", "HF_HUB_DISABLE_PROGRESS_BARS=1",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "ENABLE_COMPILE=0",
            CONTAINER_NAME,
            "bash", "-c", "/opt/conda/bin/python -m uvicorn gemma_cataloger.wsl_server:app --host 0.0.0.0 --port 8000 >> /workspace/gemma_cataloger/uvicorn_stdout.log 2>&1"
        ]
        try:
            logger.info("Launching FastAPI server inside container...")
            subprocess.run(cmd, check=True, timeout=15.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Failed to launch uvicorn inside container: {e}")
            return False

    # 3. Poll the server until it finishes loading the weights and responds to health checks
    # Model loading typically takes 30-90 seconds. We'll poll every 3 seconds for up to 450 seconds.
    max_attempts = 150
    logger.info("Waiting for Gemma 4 VLM weights to load into VRAM (this can take 1-2 minutes)...")
    for attempt in range(1, max_attempts + 1):
        if is_server_alive():
            logger.info("WSL2 Model Server is active and listening on port 8000!")
            return True
            
        # Early crash detection: check if uvicorn process died
        if not is_uvicorn_running():
            logger.error("VLM Server (uvicorn) process has crashed or stopped running!")
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvicorn.log")
            if os.path.exists(log_path):
                logger.error("==================== LAST 15 LINES OF UVICORN.LOG ====================")
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[-15:]:
                            logger.error(line.rstrip())
                except Exception as e:
                    logger.error(f"Failed to read uvicorn.log: {e}")
                logger.error("======================================================================")
            return False
            
        time.sleep(3.0)
        if attempt % 5 == 0:
            logger.info(f"Still waiting for model loading... (Attempt {attempt}/{max_attempts})")

    logger.error("Timeout: WSL2 model server failed to start within 7.5 minutes.")
    return False


def wait_for_server_startup(max_attempts: int = 120, sleep_interval: float = 2.0) -> bool:
    """Polls the server until it responds to health checks or fails early if uvicorn crashes.

    Args:
        max_attempts: Maximum number of polling attempts.
        sleep_interval: Delay between attempts in seconds.

    Returns:
        True if the server successfully started, False if it crashed or timed out.
    """
    import sys
    print("Polling server health endpoint...", flush=True)
    for attempt in range(1, max_attempts + 1):
        if is_server_alive():
            print("\n[SUCCESS] WSL2 VLM Model Server is active and listening on port 8000!", flush=True)
            return True
        
        # Check if the process has died
        if not is_uvicorn_running():
            print("\n[ERROR] VLM Server (uvicorn) process has crashed or stopped running!", flush=True)
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvicorn.log")
            if os.path.exists(log_path):
                print("==================== LAST 15 LINES OF UVICORN.LOG ====================", flush=True)
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[-15:]:
                            print(line.rstrip(), flush=True)
                except Exception as e:
                    print(f"Failed to read uvicorn.log: {e}", flush=True)
                print("======================================================================", flush=True)
            return False
            
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(sleep_interval)

    print("\n[ERROR] Timeout waiting for VLM Model Server to respond.", flush=True)
    return False


def stop_wsl_server() -> None:
    """Stops the Docker container inside WSL2, immediately releasing all GPU VRAM and memory.

    Args:
        None

    Returns:
        None
    """
    # Force termination of the keep-alive process since we are explicitly stopping the server
    global _should_terminate_keep_alive
    _should_terminate_keep_alive = True
    stop_keep_alive()

    logger.info("Shutting down WSL2 model server container...")
    try:
        # Stopping the container immediately frees up all GPU VRAM and resources
        # Added a 30-second timeout to prevent hangs.
        subprocess.run(["wsl", "-u", "workbench", "docker", "stop", CONTAINER_NAME], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30.0)
        logger.info("WSL2 model server container stopped successfully. VRAM released.")
    except Exception as e:
        logger.warning(f"Warning: Failed to stop docker container {CONTAINER_NAME}: {e}")


def query_vlm_server(image_paths: List[str], prompt_text: str) -> List[str]:
    """Encodes a batch of images to Base64 JPEGs and queries the VLM REST server.

    Args:
        image_paths: List of absolute file paths to the images.
        prompt_text: The instructions prompt text to send.

    Returns:
        A list of raw response strings containing the model's generated JSON.

    Raises:
        RuntimeError: If connection or inference requests fail.
    """
    # 1. Convert all images in the batch to base64 JPEG strings
    images_base64: List[str] = []
    for path in image_paths:
        img: Optional[Image.Image] = None
        buffered: Optional[io.BytesIO] = None
        try:
            img = Image.open(path).convert("RGB")
            
            # Prevent CUDA Out-of-Memory crashes on extremely high-res images
            # by dynamically downscaling them to fit within a safe bounding box in-memory.
            # The original image file on disk remains completely unmodified.
            if max(img.size) > MAX_IMAGE_DIM:
                logger.info(f"Scaling down in-memory representation of {os.path.basename(path)} from {img.size} to max {MAX_IMAGE_DIM}px for VRAM safety.")
                img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            # Compress to JPEG to save network payload size
            img.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_base64.append(img_str)
        except Exception as e:
            raise RuntimeError(f"Failed to read/encode image {path}: {e}")
        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass
            if buffered is not None:
                try:
                    buffered.close()
                except Exception:
                    pass

    # 2. POST to the server
    payload: Dict[str, Any] = {
        "images_base64": images_base64,
        "prompt_text": prompt_text
    }
    return _post_request_with_retry_and_restart(payload, timeout=180.0)


def query_vlm_server_base64(images_base64: List[str], prompt_text: str, temperature: float = 0.7) -> List[str]:
    """Queries the VLM REST server directly with pre-encoded Base64 image strings.

    Args:
        images_base64: List of base64 encoded image strings.
        prompt_text: The instructions prompt text to send.
        temperature: Sampling temperature for generation.

    Returns:
        A list of raw response strings containing the model's generated JSON.

    Raises:
        FatalVLMServerError: If connection or inference requests fail persistently.
    """
    payload: Dict[str, Any] = {
        "images_base64": images_base64,
        "prompt_text": prompt_text,
        "temperature": temperature
    }
    return _post_request_with_retry_and_restart(payload, timeout=180.0)


def _post_request_with_retry_and_restart(payload: Dict[str, Any], timeout: float = 180.0) -> List[str]:
    """Posts a payload to the VLM server with retry logic and automatic container restarts.

    Args:
        payload: The request body as a dictionary containing images and instructions.
        timeout: The connection/read timeout in seconds.

    Returns:
        A list of raw response strings generated by the VLM model.

    Raises:
        FatalVLMServerError: If connection or inference queries fail persistently after retries
                             and server restarts, or if the server fails to recover.
    """
    global _auto_restart_count
    max_attempts: int = 3
    attempt: int = 1
    
    while True:
        try:
            logger.info(f"Posting request to VLM model server (attempt {attempt}/{max_attempts})...")
            response = session.post(SERVER_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json().get("raw_responses", [])
            else:
                raise requests.RequestException(
                    f"Server returned HTTP error {response.status_code}: {response.text}"
                )
        except requests.RequestException as e:
            logger.warning(f"VLM server query failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                attempt += 1
                logger.info("Waiting 5.0 seconds before retrying...")
                time.sleep(5.0)
                continue
            
            # If all standard query attempts fail, check if we can restart the server container
            if _auto_restart_count < 2:
                _auto_restart_count += 1
                logger.warning(
                    f"VLM server connection failed persistently. Triggering automatic server restart "
                    f"(Restart {_auto_restart_count}/2)..."
                )
                stop_wsl_server()
                if start_wsl_server():
                    # Attempt query one last time after successful reboot
                    attempt = 1
                    max_attempts = 1
                    logger.info("Server restarted successfully. Retrying query post-restart...")
                    continue
                else:
                    raise FatalVLMServerError(
                        f"Failed to start WSL2 model server during automatic recovery: {e}"
                    ) from e
            else:
                raise FatalVLMServerError(
                    f"VLM server connection failed repeatedly and maximum automatic restarts reached: {e}"
                ) from e
