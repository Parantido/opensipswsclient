"""
Test modules for OpenSIPsCall library

This package contains test cases and example code for the OpenSIPsCall library.
Run tests using pytest from the project root:

    $ pytest tests/

For specific test files:

    $ pytest tests/test_opensips_client.py
"""

# Import common test fixtures and utilities
import os
import sys
import pytest
import asyncio
from pathlib import Path

# Add parent directory to path to make imports work in tests
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define common test fixtures
@pytest.fixture
def sample_config():
    """Return a sample configuration for testing"""
    return {
        "opensips": {
            "server": "wss://test.example.com:6061",
            "username": "test_user",
            "password": "test_password",
            "domain": "example.com",
            "debug_level": 1
        },
        "ELEVENLABS_API_KEY": "test_elevenlabs_key",
        "DEEPGRAM_API_KEY": "test_deepgram_key",
        "OPENAI_API_KEY": "test_openai_key"
    }

# Define async test helpers
@pytest.fixture
def event_loop():
    """Create a new event loop for each test"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
