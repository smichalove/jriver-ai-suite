# Google Python Style & Engineering Guidelines

> [!IMPORTANT]
> This document serves as the official instruction manual for any AI agent or developer contributing to this codebase on Windows and cross-platform setups. Always adhere to these guidelines before writing, modifying, or executing code.

---

## 1. System Performance & Model Guidelines

Leverage the system's hardware configuration to maximize performance where possible (e.g., using larger batches, caching, multi-processing, GPU acceleration, or more thread workers).

### Dedicated High-End Server (RTX 30XX/40XX/50XX or similar)
*   **VRAM**: 16GB - 24GB+ VRAM.
*   **Model Setup**: Local VLM/LLM server loaded in FP16 or INT4 weight-only quantization (`torchao`).
*   **Curation Guidelines**: Handles large context windows and batches of visual VLM scanning. Can run dedicated local FastAPI endpoints.

### Mid-Range Client (RTX 4070 / similar 12GB+ GPU)
*   **VRAM**: 12GB - 16GB VRAM.
*   **Model Setup**: Run standard local FastAPI/WSL2 or Ollama server.
*   **Curation Guidelines**: Safe for 8B/9B/12B parameters with standard context sizes (up to 16k tokens).

### Entry-Level / Consumer Client (GTX 1050)
*   **CPU**: Mid-range consumer CPU (4-8 thread workers)
*   **GPU**: Entry-level GPU (e.g. GTX 1050, 4GB VRAM)
*   **Model Setup**: Local text model (e.g., Gemma 2B, Llama 8B) running via Ollama.
*   **Curation Guidelines**: Consumer cards are highly capable of executing text curation, SQL query translation, and REPL operations. Thread worker pools should be set to match CPU core counts (e.g. 4-8 threads) to manage PostgreSQL connections. Large VLM batches (cover art visual analysis) should be skipped/disabled to prevent VRAM allocation overflows.

### Embedded Edge Devices (NVIDIA Jetson Nano / Orin / Super)
*   **System Layout**: ARM64 Architecture running NVIDIA JetPack (Linux).
*   **Model Setup**: Local text models (2B/8B/9B) executed via Ollama or Llama.cpp.
*   **Curation Guidelines**: These low-wattage boards (5W-15W) are highly cost-effective for dedicated 24/7 database operations and tag curation. Ensure database connections are carefully throttled (max 2-4 worker threads) to accommodate lower CPU cores and system bandwidth. All visual VLM tasks must be skipped on these devices.

### Apple Silicon Client (macOS M-series UMA)
*   **System Layout**: macOS Darwin running ARM64.
*   **Model Setup**: Local text/vision models executed via Ollama for Mac (utilizing Metal acceleration natively).
*   **Curation Guidelines**: M-series MacBooks are highly capable due to their Unified Memory Architecture (UMA), which allows them to allocate large portions of system RAM directly to GPU tasks without memory transfer bottlenecks. Parallel worker pools can be configured to match physical cores (e.g. 8-10 threads). Visual VLM scanning is supported locally without needing dedicated discrete graphics cards.
*   **RAM Constraints**:
    *   *8GB RAM (Minimum)*: Only run small 2B models (e.g. Gemma 2B) for database queries.
    *   *16GB RAM (Recommended)*: Safe for 8B/9B text models (e.g. Llama 3 8B, Gemma 9B) with normal parallel threads.
    *   *24GB+ UMA RAM*: Required to execute visual cover art vision scans (VLMs) locally alongside desktop apps.

*   **I/O vs GPU Inference Bottlenecks**: Do not overestimate I/O bottlenecks during local VLM batch inference. Because model inference on high-end GPUs takes significantly longer than reading and base64-encoding images on a high-speed CPU/SSD, CPU pre-processing finishes well before the GPU is ready. Consequently, scaling CPU threads/workers (e.g., raising `--max-workers` beyond the default 8) yields no performance speedup. Focus optimization efforts on model batch sizes and memory caching rather than I/O parallelism.
*   **Model Confirmation**: If you feel a need to change models, this MUST be confirmed and approved by the user prior to making any changes.

---

## 2. Docstrings and Documentation

All classes, methods, and functions must have docstrings. Docstrings should follow the Google Python Style Guide format.

### Mandatory Documentation Sync
*   **Keep all Markdown files up to date**: Always audit and update all `.md` files in the repository (such as `README.md` and `planning.md`) whenever codebase functionality, configurations, parameters, or endpoints are modified. This prevents documentation drift and ensures subsequent development steps are based on accurate system references.

### Function and Method Docstrings
Every function must include:
1. A single-line summary of what the function does (ended with a period).
2. A detailed description (optional, if the function is complex).
3. An `Args:` section detailing all parameters, their types, and roles.
4. A `Returns:` section describing the return value and its type.
5. A `Raises:` section detailing any exceptions raised by the function (if applicable).

**Example:**
```python
def calculate_tempo_offset(bpm: float, multiplier: float) -> float:
    """Calculates the adjusted tempo based on a modifier.

    Args:
        bpm: The baseline tempo in Beats Per Minute (BPM).
        multiplier: The scale factor to apply to the tempo.

    Returns:
        The adjusted tempo value as a float.

    Raises:
        ValueError: If the input bpm or multiplier is negative.
    """
    if bpm < 0 or multiplier < 0:
        raise ValueError("Inputs must be non-negative.")
    return bpm * multiplier
```

### Class Docstrings
Classes must include:
1. A summary of the class's purpose.
2. An `Attributes:` section detailing all public attributes and their types.

**Example:**
```python
class TrackMetronome:
    """Manages metronome clicks and timing intervals for a song.

    Attributes:
        bpm: The current tempo in Beats Per Minute.
        time_signature: String representing the signature (e.g. '4/4').
    """
    def __init__(self, bpm: float, time_signature: str) -> None:
        self.bpm = bpm
        self.time_signature = time_signature
```

---

## 3. Type Annotations (Strict Typing)

*   All function signatures must use strict PEP 484 type annotations for parameters and return types.
*   Use `None` explicitly as the return type for functions that do not return a value.
*   Import standard generic typing types from the `typing` module (e.g. `List`, `Dict`, `Tuple`, `Optional`, `Any`).
*   Annotate complex local variable declarations if their types are not immediately obvious.

**Example:**
```python
from typing import List, Optional

def get_track_titles(file_paths: List[str]) -> List[str]:
    titles: List[str] = []
    for path in file_paths:
        title: Optional[str] = extract_title(path)
        if title:
            titles.append(title)
    return titles
```

---

## 4. Comments and Code Explanation

*   **Explain "Why", Not "What":** Code should be self-documenting. Use comments to explain the design decisions, assumptions, performance optimizations, or constraints.
*   **Inline Comments:** Use two spaces before inline comments. Comments should start with a `#` and a single space.
*   **Block Comments:** Use block comments to explain non-obvious logic steps.

**Example:**
```python
# We use a ThreadPoolExecutor here to exploit the Ryzen 9's 32 threads.
# Running network I/O in parallel cuts GCS upload times dramatically.
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(process_file, files)
```

---

## 5. Prompts Isolation

*   **No Embedded Prompts**: Prompts, system instructions, and schemas must never be hardcoded as string literals in python scripts.
*   **External Text Files**: All prompts must be stored in dedicated external `.txt` files (e.g., `prompt.txt`, `db_prompt.txt`, `hit_and_run_prompt.txt`) and loaded dynamically at runtime right before invoking the model to allow operator adjustments in real-time.

---

## 6. Database REPL Client Guidelines

When writing or maintaining interactive database querying clients (like `db_chat_repl.py`):
*   **Read-Only Safety**: Database connections for executing VLM-generated SQL queries must be opened in read-only mode (`mode=ro`) to prevent mutation of the catalog database.
*   **Linear Context History**: Retain the complete conversational turns (including intermediate assistant tool calls and `TOOL RESULT:` message structures) in the chat history queue to preserve context across multiple conversation turns.
*   **Compact Model Feeds**: Present tool result data to the model in standard Markdown table format (with single backslashes `\`) to maximize token efficiency and prevent JSON double-backslash `\\` path confusion.
*   **Truncation Guardrails**: Limit the total return (e.g., slicing query result lists to a maximum of 50 rows) to protect context window safety while maintaining full description cell values.

---

## 7. Multi-Server & VLM Execution Guidelines

When working with multi-server configurations or remote VLM deployment:
*   **Daemon Isolation**: Always keep the remote server execution scripts isolated (e.g., `remote_server.py`) to prevent pollution of local pipelines.
*   **Dynamic Connectivity**: Implement robust fallback mechanisms that dynamically probe whether remote endpoints (e.g., WSL2 or remote hosts on port `8000`) are online before submitting batches.
*   **WSL Environment Settings**: To prevent connection reset issues under WSL2 when idle, configure the host's `.wslconfig` with `vmIdleTimeout=-1`.
*   **Synchronized Server Logic & Deployment**: Ensure all logic changes, prompt structures, or parameter modifications applied to `wsl_server.py` are mirrored exactly in `remote_server.py`. Deploy updates using SCP. When restarting remote servers, always perform a forceful process termination (`pkill -9 -f uvicorn`) and verify the old process ID is dead before launching a new instance, as soft kills can fail to terminate cached memory states.
*   **Verify Local Source Before Deployment**: Before executing `scp` deployment commands or restarting the remote server, verify that the local source file (`remote_server.py`) has been fully updated to match all modifications in `wsl_server.py`. Do not assume the files are synchronized or that a previous deployment was complete without checking the local parameters (such as `REPETITION_PENALTY` and offline environment flags).

---

## 8. VLM Prompt Structure & Generation Safety (Lessons Learned)

To prevent severe generation regressions, hallucination loops, or the model parroting few-shot examples verbatim, strictly adhere to the following rules:

*   **Explicit Role Separation (`system` vs `user`)**:
    *   Never combine the master prompt instructions/few-shot examples (`prompt.txt`) and the target image into a single `user` role prompt.
    *   **Always** isolate the master system instructions/examples in the `system` role. 
    *   The `user` role must only contain the active target image and the short command requesting the analysis (e.g., *"Analyze this image according to the archival schema..."*). This forces the model to apply the system guidelines to the new image instead of repeating the few-shot example text blocks.
*   **Generation Parameters Protection**:
    *   **Temperature**: Maintain sampling (`do_sample=True`) at the stable default of `0.7`. Never unilaterally lower the generation temperature to low values like `0.2` when utilizing complex few-shot instructions, as it makes the model highly deterministic and triggers verbatim parroting of the context.
    *   **Repetition Penalty**: Keep the repetition penalty configured to `1.15` in server endpoints to prevent phrase repetition loops without sacrificing output quality.

---

## 9. Staging Environment Parity

*   **Mandatory Schema & Config Parity:** Any staging, testing, or sandboxed database/environment must utilize a schema and configurations that match the active production environment exactly. Never use simplified or partial schemas for testing pipelines that will be used for production migrations or data restores.
*   **Schema Verification Before Migration:** Before dumping data from any staging/test instance for production deployment, the agent must programmatically compare the schemas (columns, indexes, constraints) and row metrics between source and destination targets to prevent silent data omission or column loss.

---

## 10. Music Curation Pipeline & Ingestion Guidelines

Guidelines for optimizing and refactoring the music cataloging database structures:

*   **Concurrent Multi-Server Distribution**: Music directories are processed concurrently. The script maps curation requests across a ThreadPoolExecutor queue to active network endpoints (workstations, WSL2, and remote nodes).
*   **Incremental DB Commits**: Curation updates must be written and committed to PostgreSQL folder-by-folder inside the worker thread immediately. Never delay commits until the end of the entire loop.
*   **Prevent Repeat Sweeps (Gaps Loop)**: Ensure offline heuristic fallbacks map unresolved tags to `'Unresolved Artist'`, `'Unresolved Album'`, and `'Unresolved Genre'` instead of `None` or `'Unknown'`, permanently removing them from future gap sweep queries.
*   **Immediate Fallback**: If an LLM node fails, trigger and commit the offline split heuristic on the spot inside the thread, marking the folder as resolved and continuing.
*   **task_done() try-finally safety**: In Python `try...finally` structures, `finally` always executes even when `continue` is called inside `try`. Never call `task_done()` manually inside loop condition skips if it is already present in `finally`.
*   **Parallel File Ingest & ExifTool Execution**: Filesystem crawling, directory scanning, and JRiver/ExifTool/mutagen metadata parsing must always be executed in parallel using `ThreadPoolExecutor` and `--max-workers`. Do not implement sequential parsing or argue that parallel ExifTool execution causes disk thrashing or is unsafe; parallel execution is highly performant, stable, and tested on this server's NVMe storage array and Ryzen 9 CPU architecture.
*   **Default to Parallelism for All Code**: For all filesystem, network I/O, database, or API operations, always default to multithreaded or parallel designs (utilizing `ThreadPoolExecutor` or `ProcessPoolExecutor` with customizable worker arguments) to maximize throughput, rather than using slow sequential loops.
*   **Incremental Batch Commits & Memory Safety**: For large-scale data ingestion or scanning operations, never buffer the entire workload in memory to execute a single transaction at the end. Always flush and commit records incrementally (e.g., in chunks of 1,000 files or folder-by-folder inside parallel threads) and clear the buffer to manage memory limits and guarantee progress recovery.




