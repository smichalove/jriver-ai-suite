"""Modular JRiver Sidecar XML and Audio File Metadata Parser Test.

Purpose:
    This script acts as a test validator to recursively search a target directory
    for JRiver Media Center sidecar XML files (*_JRSidecar.xml), parse their XML tag contents,
    extract technical properties from the corresponding media files using ExifTool,
    and merge the data structures into a single unified record.

Architecture and Mechanics:
    1. XML Parsing: Loads and parses JRiver's proprietary MPL XML schemas, mapping all
       dynamic `<Field Name="...">` properties into a flat key-value dictionary.
    2. ExifTool Integration: Invokes ExifTool to extract technical stream properties
       (bitrate, sample rate, channels, file type, duration) directly from media files.
    3. Structural Merging: Pairs the sidecar XML fields with their corresponding media
       file metadata to create an empirical database-ready structure.

Execution Modes:
    - Command Line:
      python test_music_meta_parser.py --dir <music_album_folder>
"""

import os
import sys
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

# Ensure standard UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Default path to ExifTool binary
DEFAULT_EXIFTOOL_PATH: str = r"H:\Wan_project\exiftool\exiftool.exe"


def parse_jr_sidecar(xml_path: str) -> Dict[str, str]:
    """Parses a JRiver Media Center XML sidecar into a flat key-value dictionary.

    Args:
        xml_path: The absolute filesystem path to the XML sidecar.

    Returns:
        A dictionary mapping the XML Field Name attributes to their text content.
    """
    metadata: Dict[str, str] = {}
    if not os.path.exists(xml_path):
        return metadata

    try:
        tree: ET.ElementTree = ET.parse(xml_path)
        root: ET.Element = tree.getroot()
        # Locate the <Item> block and extract each <Field> element
        for item in root.findall(".//Item"):
            for field in item.findall("Field"):
                name: Optional[str] = field.get("Name")
                text: Optional[str] = field.text
                if name:
                    metadata[name] = text if text is not None else ""
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to parse XML sidecar {xml_path}: {e}\n")

    return metadata


def extract_media_metadata(exiftool_path: str, media_path: str) -> Dict[str, str]:
    """Extracts technical file properties from a media file using ExifTool.

    Args:
        exiftool_path: The filesystem path to the ExifTool executable.
        media_path: The filesystem path to the target media file.

    Returns:
        A dictionary of technical properties extracted from the file.
    """
    metadata: Dict[str, str] = {}
    if not os.path.exists(media_path):
        return metadata

    cmd: List[str] = [
        exiftool_path,
        "-json",
        "-n",
        "-charset", "filename=utf8",
        "-Duration",
        "-AudioBitrate",
        "-Bitrate",
        "-SampleRate",
        "-AudioChannels",
        "-Channels",
        "-FileType",
        "-FileSize",
        media_path
    ]

    try:
        # Run ExifTool synchronously with a timeout
        res: subprocess.CompletedProcess[bytes] = subprocess.run(
            cmd, capture_output=True, timeout=30.0, check=False
        )
        if res.stdout:
            raw_data: List[Dict[str, str]] = json.loads(res.stdout)
            if raw_data and isinstance(raw_data, list):
                # The output is a list containing a single file metadata dictionary
                return raw_data[0]
    except Exception as e:
        sys.stderr.write(f"[ERROR] ExifTool extraction failed for {media_path}: {e}\n")

    return metadata


def test_directory_parsing(exiftool_path: str, target_dir: str) -> None:
    """Walks the target directory, pairs XML sidecars with media files, and merges them.

    Args:
        exiftool_path: Path to the ExifTool binary.
        target_dir: Path to the directory to test parse.
    """
    print(f"Scanning target directory: {target_dir}\n")
    if not os.path.exists(target_dir):
        print(f"[ERROR] Target directory does not exist: {target_dir}")
        return

    # Walk directory to find XML files
    xml_files: List[str] = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith("_JRSidecar.xml"):
                xml_files.append(os.path.join(root, file))

    if not xml_files:
        print("No JRiver sidecar (*_JRSidecar.xml) files found in this directory.")
        return

    print(f"Found {len(xml_files)} JRiver sidecar files. Merging with file metadata...\n")

    for xml_path in xml_files:
        print(f"Processing sidecar: {os.path.basename(xml_path)}")
        xml_metadata: Dict[str, str] = parse_jr_sidecar(xml_path)

        # Retrieve media filename from the JRiver XML or fallback to the name match
        media_path: str = xml_metadata.get("Filename", "")
        if not media_path or not os.path.exists(media_path):
            # Fallback check: guess filename by replacing _JRsidecar.xml with original extension
            # JRiver usually names XMLs as: <filename>_<ext>_JRSidecar.xml
            base_dir: str = os.path.dirname(xml_path)
            xml_name: str = os.path.basename(xml_path)
            # Remove '_JRSidecar.xml'
            cleaned_name: str = xml_name[:-14]
            # Split off the extension (e.g. Jimi Hendrix - Live_mpg -> mpg is extension)
            if "_" in cleaned_name:
                parts: List[str] = cleaned_name.rsplit("_", 1)
                guessed_media_name: str = f"{parts[0]}.{parts[1]}"
                media_path = os.path.join(base_dir, guessed_media_name)

        if not os.path.exists(media_path):
            print(f"  [WARN] Associated media file not found: {media_path}")
            continue

        print(f"  Associated media found: {os.path.basename(media_path)}")
        file_metadata: Dict[str, str] = extract_media_metadata(exiftool_path, media_path)

        # Merge structural dictionaries
        merged_metadata: Dict[str, str] = {**xml_metadata, **file_metadata}

        print("  === Merged Data Structure ===")
        # Print a subset of interesting fields to keep output readable
        keys_to_print: List[str] = [
            "Filename", "Name", "Artist", "Album", "Genre", 
            "Rating", "Duration", "Bitrate", "Sample Rate", 
            "Channels", "FileType", "FileSize"
        ]
        for key in keys_to_print:
            val: str = merged_metadata.get(key, "N/A")
            print(f"    - {key}: {val}")
        print("-" * 60)


def main() -> None:
    """Main CLI entry point parsing arguments and invoking the parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Verify JRiver XML and media file metadata merging."
    )
    parser.add_argument(
        "--dir", 
        required=True, 
        help="Directory containing the music files and sidecar XMLs."
    )
    parser.add_argument(
        "--exiftool", 
        default=DEFAULT_EXIFTOOL_PATH, 
        help="Path to the ExifTool executable."
    )
    args: argparse.Namespace = parser.parse_args()

    test_directory_parsing(args.exiftool, args.dir)


if __name__ == "__main__":
    main()
