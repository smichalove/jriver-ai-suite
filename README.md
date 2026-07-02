# 🎵 JRiver AI Suite

A high-performance, private, local AI companion suite for [JRiver Media Center](https://www.jriver.com/) (available for purchase [here](https://www.jriver.com/purchase.shtml)). It enables lightning-fast audio/video metadata indexing, natural-language database querying (translating English to SQL), automatic spelling corrections, and real-time playback control (playing, queuing, and adding tracks to your active zone).

> **Developer Note:** JRiver Media Center is a best-in-class, highly performant media management platform for music and images, and I have been using it since 2005. This suite was built to bridge the gap between JRiver's robust media engine and modern local Generative AI.

> [!IMPORTANT]
> **AI Coding Assistants & Developers:** Before writing code, refactoring, or executing commands in this repository, please review the strict style guidelines, type annotations, and performance rules in [agents.md](agents.md).

---

## ⚡ Key Features

*   🚀 **High-Speed Parallel Ingestion**: Crawls and indexes local audio libraries (FLAC, WAV, M4A, APE) and video sidecars recursively using parallel `mutagen` extraction. Scans and commits **60,000+ tracks to PostgreSQL in under 90 seconds** (800+ tracks/sec).
    *   *Note on SQLite Concurrency:* While SQLite is supported (configured via `DB_BACKEND="sqlite"`), executing highly parallelized multi-threaded ingestion (e.g., 20+ threads) against SQLite can trigger database write-lock contention ("database is locked" errors). A local PostgreSQL instance is strongly recommended to handle these concurrent threads smoothly.
*   💬 **Natural Language Database Chat**: A local REPL client that translates conversational questions (e.g. *"play some upbeat acoustic tracks by Imogen Heap from 2015"*) into optimized SQL queries on the fly.
*   🔀 **Direct JRiver Integration (MCWS)**: Sends playback and queue commands directly to JRiver Media Center's Web Service (Zone 0) in-memory, ensuring zero clutter on your filesystem.
*   🧠 **Flexible Tag Curation & Normalization**: Automatically normalizes messy tag gaps (Artist, Album, Genre) and flags non-music folders dynamically using local LLM models.

---

## 💻 Hardware Pathways & Scaling

You do **not** need a high-end AI workstation or data center to run this suite. The architecture scales gracefully depending on your system's GPU:

### 1. 🟢 Entry-Level / Consumer GPUs (GTX 1050 / 4GB+ VRAM)
*   **Capability**: Full natural-language database querying, interactive REPL chat, spelling corrections, and track tag curation.
*   **Setup**: Run standard **Ollama** locally on Windows.
*   **Recommended Models**: Quantized Llama 3 8B or Gemma 2 2B / 9B.
*   **Note**: Text-based SQL translation and metadata cleaning run extremely fast on lightweight consumer cards (e.g., GTX 1050, RTX 2060).

### 2. 🔵 Enthusiast / High-End GPUs (RTX 3090 / 4090 / 5080 / 12GB+ VRAM)
*   **Capability**: Adds visual VLM processing for album cover art and booklets.
*   **Setup**: Run a local/remote FastAPI server inside WSL2 running vision models (e.g., Gemma 4-it-q4).
*   **Usage**: Generates archival, descriptive paragraphs of cover artwork designs, saves them to PostgreSQL, and writes the descriptions back to the image file EXIF headers in parallel.

### 3. 🧠 Embedded Edge Devices (NVIDIA Jetson Nano / Orin / Super)
*   **Capability**: Full natural-language database chat, SQL translation, and metadata curation.
*   **Setup**: Run standard Ollama or Llama.cpp natively on Ubuntu Linux (the operating system of choice for these edge configurations, e.g. via NVIDIA JetPack).
*   **Benefit**: Extremely cost-effective and power-efficient alternative. Ideal for running 24/7 background curation and database operations on low-wattage setups (5W-15W) rather than powering a high-draw workstation GPU.

### 4. 🍎 macOS Platforms (Apple Silicon M1/M2/M3/M4 & Intel Macs)
*   **Capability**: Full natural-language chat querying, spelling fallbacks, and tag curation. Apple Silicon (M-series) Macs can also run local vision model (VLM) cover scanning.
*   **Setup**: Run Ollama natively on macOS (with automatic Metal GPU acceleration).
*   **Hardware Constraints**:
    *   *8GB Unified RAM (Minimum)*: Suitable for running 2B parameter models (e.g. Gemma 2B) for SQL querying.
    *   *16GB Unified RAM (Recommended)*: Required for comfortable 8B/9B text models (e.g. Llama 3 8B, Gemma 9B) to prevent performance lag from memory swapping.
    *   *24GB+ Unified RAM (UMA)*: Required to run local visual vision models (VLMs) alongside JRiver Media Center and standard desktop apps.
*   **Benefit**: Apple Silicon's **Unified Memory Architecture (UMA)** allows the CPU and GPU to share the same high-speed system memory pool. This lets even a standard MacBook run 8B/9B text and vision models extremely fast without VRAM transfer bottlenecks, making it a highly portable and quiet companion for JRiver curation.

### 5. 🌐 Local Distributed Multi-Node Architecture
*   **Setup**: Run model servers on local workstations, WSL2 instances, or dedicated Linux servers/nodes across your home network.
*   **Capability**: The ingestion orchestrator natively supports distributed threading, mapping metadata curation jobs concurrently across multiple active HTTP/Ollama/VLM endpoints. This multi-nodal setup enables highly performant distributed threading to maximize curation throughput by leveraging all idle computing hardware in your network.

> [!TIP]
> **Cloud Gemini Alternative**: If you prefer cloud-based API inference instead of running models locally, Google Gemini API endpoints can easily be integrated into this pipeline. For a reference implementation and code patterns, see the [Gemini Photo Batch Workflow](https://github.com/smichalove/Gemini_Photo_Batch_Workflow) repository.

---

## ⚙️ Quick Start

### 1. Installation & Setup
Clone the repository and install the dependencies in a virtual environment:
```bash
# Clone this repository and navigate into it
git clone <your-repository-url>
cd jriver-AI-suite

# Create and activate a virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Database Provisioning
Configure a local PostgreSQL database instance:
```sql
CREATE DATABASE photo_catalog;
```
Ensure you have the connection details ready.

### 3. Configuration (`.env`)
Create a `.env` file under `auth/.env` with your database credentials:
```ini
DB_BACKEND="postgresql"
DB_HOST="localhost"
DB_PORT="5432"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_NAME="photo_catalog"

# JRiver MCWS Config
JRIVER_PORT="52198"
JRIVER_ZONE="0"
```

### 4. Run Ingestion
Initialize your music library database index:
```bash
run_music_ingest.bat
```
This recursive walk will skip already-indexed tracks, process new files in parallel, and commit to the database in chunks of 1,000.

### 5. Open the REPL Chat & Control JRiver
Start the interactive chat interface:
```bash
run_db_chat_pg.bat
```
Now you can search, query, and play music:
*   `find all tracks by Dire Straits`
*   `/play` (plays all matching tracks)
*   `/queue 5` (adds the 5th track in the list to the queue)

---

## 🧪 Verification & Tests

Run the comprehensive unit test suite to verify XML parsing, ingestion database workers, curation logic, and MCWS playback commands:
```powershell
# Set backend to sqlite for clean isolated mock tests, then run discover inside tests/
$env:DB_BACKEND='sqlite'; python -m unittest discover -s tests
```

---

## 📖 Appendix: Example Chat Session

Below is a typical interactive CLI session with the JRiver AI Suite REPL. It demonstrates SQL query translation, large-result handling, automatic queue integration (`/queue`), and case-insensitive search queries.

```text
[Remote Mode] Connecting to model server at http://127.0.0.1:11434 using model 'gemma2:9b'...
==================================================
  JRiver AI Suite - Database Chat & Playback
==================================================
Instructions:
  * Ask questions about your music library in plain English.
  * Use '/play' or '/queue' to control JRiver Media Center playback.
  * Type 'exit' to quit.
==================================================

Prompt > find all tracks by Dire Straits from the 80s
[Assistant]: I will query the database for tracks by Dire Straits released in the 1980s.
<tool_call>{"tool": "query_db", "sql": "SELECT file_path, title, artist, album FROM music_tracks WHERE artist ILIKE '%Dire Straits%' AND year BETWEEN 1980 AND 1989 LIMIT 10"}</tool_call>

[Results]:
1. "D:\Users\steven\Music\Dire Straits\Brothers In Arms\01 - So Far Away.flac" - So Far Away (Artist: Dire Straits, Album: Brothers In Arms)
2. "D:\Users\steven\Music\Dire Straits\Brothers In Arms\02 - Money For Nothing.flac" - Money For Nothing (Artist: Dire Straits, Album: Brothers In Arms)
3. "D:\Users\steven\Music\Dire Straits\Brothers In Arms\03 - Walk Of Life.flac" - Walk Of Life (Artist: Dire Straits, Album: Brothers In Arms)

Prompt > /play 2
[JRiver Play] Preparing to play track index 2 on JRiver Media Center...
[JRiver Play] ✅ Successfully queued & started playback in JRiver!

Prompt > find tracks from 2010 to 2015
[Assistant]: I will query the database for tracks released between 2010 and 2015.
<tool_call>{"tool": "query_db", "sql": "SELECT file_path, title, artist, album FROM music_tracks WHERE year BETWEEN 2010 AND 2015 LIMIT 300"}</tool_call>

[Results]:
Since this search returned 300 results, the full list of file paths and descriptions has already been printed to your terminal. Please refer to that list for the complete collection of tracks found.

Prompt > /queue
[JRiver Queue] Preparing to queue 299 track(s) on JRiver Media Center...
[JRiver Queue] ✅ Successfully queued 299 track(s) in JRiver!

Prompt > find tracks by like Heap no limit
[Assistant]: I will query the database for all tracks with an artist or album matching "Heap" with no result limit.
<tool_call>{"tool": "query_db", "sql": "SELECT file_path, title, artist, album FROM music_tracks WHERE artist ILIKE '%Heap%' OR album ILIKE '%Heap%'"}</tool_call>

[Results]:
1. "D:\Users\steven\Music\Imogen Heap\Speak For Yourself\01 - Headlock.flac" - Headlock (Artist: Imogen Heap, Album: Speak For Yourself)
2. "D:\Users\steven\Music\Imogen Heap\Speak For Yourself\02 - Hide And Seek.flac" - Hide And Seek (Artist: Imogen Heap, Album: Speak For Yourself)
```
