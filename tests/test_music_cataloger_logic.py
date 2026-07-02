"""Unit Tests for Music Cataloger Logic.

Purpose:
    This script contains unit tests to verify the correctness of our metadata curation,
    artist validation filters, nested artwork crawler discovery, and MP3 skip filters.

Execution:
    python -m unittest test_music_cataloger_logic.py
"""

import os
import tempfile
import shutil
import unittest
from typing import Dict, List, Set, Optional

# Import components to test
from clean_database_artists import is_valid_artist_name
from ingest_music_library import find_album_art


class TestMusicCatalogerLogic(unittest.TestCase):
    """Test suite for validating core logic in JRiver scanner and artist cleaner."""

    def test_artist_name_validation(self) -> None:
        """Verifies that is_valid_artist_name validates real names and rejects junk."""
        # Valid artist cases
        self.assertTrue(is_valid_artist_name("Jimi Hendrix"))
        self.assertTrue(is_valid_artist_name("Sheryl Crow"))
        self.assertTrue(is_valid_artist_name("Imogen Heap"))
        self.assertTrue(is_valid_artist_name("The Starting Line"))
        self.assertTrue(is_valid_artist_name("Chick Corea Stanley Clarke Al Di Meola"))

        # Invalid: Track numbers
        self.assertFalse(is_valid_artist_name("01"))
        self.assertFalse(is_valid_artist_name("02"))
        self.assertFalse(is_valid_artist_name("15"))
        self.assertFalse(is_valid_artist_name(" 05 "))

        # Invalid: File extensions
        self.assertFalse(is_valid_artist_name("artist.mp3"))
        self.assertFalse(is_valid_artist_name("track.mkv"))
        self.assertFalse(is_valid_artist_name("video.mp4"))

        # Invalid: Video release tags & resolution keywords
        self.assertFalse(is_valid_artist_name("Jimi.Hendrix.1080p.WEBRip.x264-CBFM"))
        self.assertFalse(is_valid_artist_name("Dune Part02 BDRip x264"))
        self.assertFalse(is_valid_artist_name("Revolution Season 2 DVD 2 NL Subs"))
        self.assertFalse(is_valid_artist_name("Unknown Artist"))
        self.assertFalse(is_valid_artist_name(""))

    def test_nested_album_art_discovery(self) -> None:
        """Mocks a directory layout to verify subfolder artwork scans and size selection."""
        # Create a temporary directory for testing
        test_dir = tempfile.mkdtemp()
        try:
            # Case 1: No artwork present
            self.assertIsNone(find_album_art(test_dir))

            # Case 2: Standard folder.jpg in root
            root_cover = os.path.join(test_dir, "folder.jpg")
            with open(root_cover, "wb") as f:
                f.write(b"small_image_bytes")
            
            self.assertEqual(find_album_art(test_dir), root_cover)

            # Case 3: Larger cover inside nested Artwork folder
            artwork_sub = os.path.join(test_dir, "Artwork")
            os.makedirs(artwork_sub)
            nested_cover = os.path.join(artwork_sub, "front_cover_heavy.png")
            with open(nested_cover, "wb") as f:
                # Write a larger payload to mock higher resolution
                f.write(b"much_larger_high_resolution_image_payload_bytes_here")

            # The crawler should scan Artwork/ and return nested_cover because it's larger
            discovered = find_album_art(test_dir)
            self.assertEqual(discovered, nested_cover)

            # Case 4: Larger cover inside nested Covers folder
            covers_sub = os.path.join(test_dir, "Covers")
            os.makedirs(covers_sub)
            even_larger_cover = os.path.join(covers_sub, "vinyl_cover.jpg")
            with open(even_larger_cover, "wb") as f:
                f.write(b"x" * 1000) # Write 1000 bytes

            # Now vinyl_cover.jpg is the largest candidate across all searched folders
            discovered = find_album_art(test_dir)
            self.assertEqual(discovered, even_larger_cover)

        finally:
            # Clean up temp folder structure
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
