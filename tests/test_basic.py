"""Basic tests for the withsecure ML Pipeline."""

import os

def test_project_structure():
    """Test that required directories exist."""
    required_dirs = ['src', 'src/config', 'src/model', 'src/model/pipeline']
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Required directory {dir_path} does not exist"

