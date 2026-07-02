"""Unit Test Suite for Multi-Server VLM Cataloging.

Purpose:
    This module implements a comprehensive unit test suite to verify the VLMServerConnection
    and multi-server parallel batch processing features of the cataloging pipeline.

Architecture and Mechanics:
    1. Mocking Framework: Uses unittest.mock to stub out network calls (requests.get/post)
       and file loads.
    2. Target Subsystems Tested:
       - VLMServerConnection Lifecycle: Verifies initialization and URL generation.
       - Health Probing: Mocks request status codes to test if is_alive returns True/False correctly.
       - Query Dispatching: Verifies that payload payloads are serialized correctly and responses returned.
       - Concurrent Queue Processing: Simulates multiple worker threads pulling items from a queue and calling mock servers in parallel.

Execution Modes:
    - Test Runner: Run using python's unittest runner.
      Command:
        python -m unittest gemma_cataloger/test_multi_server.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# Ensure parent directory is in sys.path to allow importing from gemma_cataloger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsl_client import VLMServerConnection
from describe_photos import process_batches


class TestMultiServerVLM(unittest.TestCase):
    """Unit test suite for the VLMServerConnection and multi-server batch processing."""

    def test_connection_init(self) -> None:
        """Tests that VLMServerConnection initializes correctly and generates accurate URLs.

        Args:
            None

        Returns:
            None
        """
        conn = VLMServerConnection("Test Server", "http://192.168.8.113:8000/", batch_size=3)
        self.assertEqual(conn.name, "Test Server")
        self.assertEqual(conn.base_url, "http://192.168.8.113:8000")
        self.assertEqual(conn.batch_size, 3)
        self.assertEqual(conn.describe_url, "http://192.168.8.113:8000/describe")
        self.assertEqual(conn.health_url, "http://192.168.8.113:8000/docs")

    @patch("wsl_client.session.get")
    def test_connection_is_alive(self, mock_get: MagicMock) -> None:
        """Tests that is_alive returns True on success and False on network errors.

        Args:
            mock_get: Mocked requests session get method.

        Returns:
            None
        """
        conn = VLMServerConnection("Test Server", "http://localhost:8000")
        
        # Test success (200 OK)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        self.assertTrue(conn.is_alive())
        
        # Test failure (500 Error)
        mock_response.status_code = 500
        self.assertFalse(conn.is_alive())
        
        # Test connection exception
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")
        self.assertFalse(conn.is_alive())

    @patch("wsl_client.session.post")
    def test_connection_query(self, mock_post: MagicMock) -> None:
        """Tests that query formats payloads correctly and returns the VLM raw responses.

        Args:
            mock_post: Mocked requests session post method.

        Returns:
            None
        """
        conn = VLMServerConnection("Test Server", "http://localhost:8000")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"raw_responses": ["{\"subject\": \"cat\"}"]}
        mock_post.return_value = mock_response

        res = conn.query(["base64string"], "Test Prompt", temperature=0.3)
        self.assertEqual(res, ["{\"subject\": \"cat\"}"])
        
        # Verify call parameters
        mock_post.assert_called_once_with(
            "http://localhost:8000/describe",
            json={
                "images_base64": ["base64string"],
                "prompt_text": "Test Prompt",
                "temperature": 0.3
            },
            timeout=600.0
        )

    @patch("describe_photos.load_and_encode_image")
    @patch("describe_photos.save_results")
    @patch("describe_photos.save_results_to_sqlite")
    def test_process_batches_parallel(self, mock_sqlite: MagicMock, mock_json: MagicMock, mock_load: MagicMock) -> None:
        """Tests process_batches handles distribution across multiple active servers successfully.

        Args:
            mock_sqlite: Mocked SQLite save function.
            mock_json: Mocked JSON save function.
            mock_load: Mocked image loading function.

        Returns:
            None
        """
        mock_load.return_value = "fake_b64"
        
        # Create mock servers
        server1 = MagicMock(spec=VLMServerConnection)
        server1.name = "Server 1"
        server1.batch_size = 2
        server1.query.return_value = [
            "{\"primary_subject\": \"sky\", \"environment\": \"outdoors\", \"suggested_tags\": []}",
            "{\"primary_subject\": \"ground\", \"environment\": \"outdoors\", \"suggested_tags\": []}"
        ]

        server2 = MagicMock(spec=VLMServerConnection)
        server2.name = "Server 2"
        server2.batch_size = 1
        server2.query.return_value = [
            "{\"primary_subject\": \"tree\", \"environment\": \"outdoors\", \"suggested_tags\": []}"
        ]

        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        results: List[Dict[str, Any]] = []

        # Run parallel batch processing
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data="Test prompt")):
            process_batches(
                image_paths=image_paths,
                results=results,
                prompt_path="prompt.txt",
                active_servers=[server1, server2],
                max_workers=2,
                output_json="output.json",
                db_path="test.db",
                embed_exif=False,
                no_json_update=False,
                temperature=0.2
            )

        # Assert results were saved
        self.assertTrue(len(results) > 0)
        self.assertTrue(mock_json.called)
        self.assertTrue(mock_sqlite.called)


if __name__ == "__main__":
    unittest.main()
