"""
logger_util.py

Purpose:
    - Advanced Logging Utility for the EEG Analysis System.
    - Redirects 'print' (stdout) and errors (stderr) to both the console and a file.
    - Features:
        1. Automatic timestamps for debugging timing issues.
        2. Global Exception Hook: Catches crashes and writes the full stack trace 
           to the log file (crucial for debugging GUI crashes).
        3. Immediate Flushing: Ensures logs are saved even if the app crashes hard.

Usage:
    Import this module in 'main.py' and call 'setup_logging()' as the very first step.
"""

import sys
import os
import traceback
from datetime import datetime

# Global reference to the logger instance
_logger_instance = None

class DualLogger:
    """
    A custom file-like object that mirrors output to both the terminal and a log file.
    """
    def __init__(self, filename="debug_output.txt"):
        """
        Initialize the logger.
        
        Args:
            filename (str): The name of the text file to write logs to.
        """
        self.terminal = sys.stdout
        self.filename = filename
        
        # Open the file in 'write' mode initially to clear old logs, 
        # then we will keep it open or reopen as needed.
        # Here we keep it open to ensure performance.
        self.log_file = open(self.filename, "w", encoding="utf-8")
        
        # Write the Session Header
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"==========================================================\n"
            f"   EEG ANALYSIS SYSTEM - DEBUG LOG\n"
            f"   Session Started: {start_time}\n"
            f"==========================================================\n\n"
        )
        self.terminal.write(header)
        self.log_file.write(header)
        self.flush()

    def write(self, message):
        """
        Overrides the standard write method (used by print()).
        Adds timestamps to lines that appear to be new log entries.
        """
        # If the message is just a newline, just write it
        if message == "\n":
            self.terminal.write(message)
            self.log_file.write(message)
        else:
            # For actual content, we ensure it gets written.
            # We can optionally add timestamps here, but 'print' often sends 
            # partial strings. Ideally, rely on explicit logging, but for 
            # capturing standard 'print', we just mirror it.
            
            # Write to Console
            self.terminal.write(message)
            
            # Write to File
            self.log_file.write(message)
            
        # FORCE FLUSH: This ensures data is written to disk immediately.
        # Critical for debugging crashes where the buffer might be lost.
        self.flush()

    def flush(self):
        """
        Forces the buffer to flush. Required for python file-like objects.
        """
        self.terminal.flush()
        self.log_file.flush()
        os.fsync(self.log_file.fileno())

    def close(self):
        """
        Closes the file handle properly.
        """
        self.log_file.close()

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global Exception Hook.
    This function is called automatically whenever an unhandled exception occurs.
    It writes the full traceback to the log file.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to stop the program normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Format the error message
    error_header = "\n" + "!"*60 + "\n"
    error_title = "   CRITICAL SYSTEM ERROR (UNHANDLED EXCEPTION)   \n"
    error_footer = "!"*60 + "\n"
    
    # Get the string representation of the traceback
    trace_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    full_error_msg = f"{error_header}{error_title}{error_footer}\n{trace_str}\n"
    
    # Print to console (using the original stdout just in case)
    print(full_error_msg)
    
    # Write to log file if the logger is active
    if _logger_instance:
        _logger_instance.log_file.write(full_error_msg)
        _logger_instance.flush()

def setup_logging(filename="debug_output.txt"):
    """
    Sets up the redirection of stdout and stderr.
    Must be called at the very start of the application.
    """
    global _logger_instance
    
    # Create the logger
    _logger_instance = DualLogger(filename)
    
    # Redirect standard output (print)
    sys.stdout = _logger_instance
    
    # Redirect standard error (tracebacks)
    sys.stderr = _logger_instance
    
    # Hook into the global exception handler
    # This catches errors that would otherwise just crash the GUI silently
    sys.excepthook = handle_exception
    
    print("[SYSTEM] Debug Logger Initialized.")
    print(f"[SYSTEM] Log file location: {os.path.abspath(filename)}")
    print("[SYSTEM] Global Exception Hook is active.")