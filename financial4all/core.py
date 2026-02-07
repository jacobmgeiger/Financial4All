# financial4all/core.py
"""
Core utilities and base classes for Financial4All.

This module provides common utilities, base classes, and helper functions
used throughout the package.
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path

from financial4all.config import get_config

# Set up logging
log = logging.getLogger("financial4all")
log.setLevel(logging.INFO)

# Create console handler if not already configured
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    log.addHandler(handler)


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    
    PyInstaller creates a temp folder and stores path in _MEIPASS.
    This function handles both development and packaged environments.
    
    Args:
        relative_path: Relative path to the resource
        
    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Development environment
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # This file is in financial4all/core.py, so go up two levels
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path to the data directory
    """
    return get_project_root() / "data"


def get_xbrl_prep_dir() -> Path:
    """
    Get the XBRL preparation directory path.
    
    Returns:
        Path to the xbrl_prep directory
    """
    return get_project_root() / "xbrl_prep"
