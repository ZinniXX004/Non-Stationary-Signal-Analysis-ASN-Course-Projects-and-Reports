"""
logger_util.py

Purpose:
    - Redirect stdout and stderr to both console and a text file.
    - Helps in debugging by persisting error messages.
"""

import sys
import os
from datetime import datetime

class DualLogger:
    def __init__(self, filename="debug_log.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")
        
        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"=== SESSION STARTED: {timestamp} ===\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush() # Ensure it writes immediately

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log_file.flush()

def setup_logging():
    # Redirect stdout and stderr
    sys.stdout = DualLogger("debug_output.txt")
    sys.stderr = sys.stdout 
    print("[SYSTEM] Logging initialized. Output saved to 'debug_output.txt'")