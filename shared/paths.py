#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# Define project root (assuming paths.py is in the project root or a config/ directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjust ".parent" depth as needed

# Directory definitions
DATA_DIR = PROJECT_ROOT / "data"

# File paths 
ANALYSIS_PARAMS = DATA_DIR / "analysis_parameters_correlation_project.csv"  
EXPERIMENTS = DATA_DIR / "experiments_correlation_project.csv"

print(ANALYSIS_PARAMS)