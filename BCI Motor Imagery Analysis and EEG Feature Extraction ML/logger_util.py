"""
logger_util.py

Purpose:
    - Advanced Logging Utility for the EEG Analysis System.
    - Redirects standard output (stdout) and error output (stderr) to both the console and a text file.
    - Features:
        1. Automatic timestamps for precise event tracking.
        2. Global Exception Hook: Catches crashes and writes the full stack trace to the log file.
        3. Immediate Flushing: Ensures logs are saved even if the application crashes abruptly.
    
Usage:
    Import this module in 'main.py' and call 'setup_logging()' before any other logic.
"""

import sys
import os
import traceback
from datetime import datetime

# Global reference to the logger instance to allow access from the exception hook
_logger_instance = None

class DualLogger:
    """
    A custom file-like object that mirrors output to both the terminal and a log file.
    It replaces sys.stdout and sys.stderr.
    """
    def __init__(self, filename="debug_output.txt"):
        """
        Initialize the logger.
        
        Args:
            filename (str): The name of the text file to write logs to.
        """
        self.terminal = sys.stdout
        self.filename = filename
        
        # Open the file in 'write' mode ('w') initially to clear old logs from previous runs.
        # We use UTF-8 encoding to support special characters if needed.
        try:
            self.log_file = open(self.filename, "w", encoding="utf-8")
        except IOError as e:
            print(f"Error opening log file: {e}")
            self.log_file = None
        
        # Write the Session Header
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"==========================================================\n"
            f"   EEG ANALYSIS SYSTEM - DEBUG LOG\n"
            f"   Session Started: {start_time}\n"
            f"==========================================================\n\n"
        )
        
        # Write header to both destinations
        self.terminal.write(header)
        if self.log_file:
            self.log_file.write(header)
            self.flush()

    def write(self, message):
        """
        Overrides the standard write method (used by print()).
        
        Args:
            message (str): The text to write.
        """
        # Write to the standard console (Terminal)
        self.terminal.write(message)
        
        # Write to the Log File
        if self.log_file:
            # We filter out standalone newline characters sometimes sent by print() 
            # to avoid excessive blank lines in the log file, but generally we mirror everything.
            self.log_file.write(message)
            
            # FORCE FLUSH: This ensures data is written to the physical disk immediately.
            # This is critical for debugging crashes where the buffer might be lost if not flushed.
            self.flush()

    def flush(self):
        """
        Forces the buffer to flush. Required for python file-like objects compatibility.
        """
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
            try:
                # Force OS to write to disk
                os.fsync(self.log_file.fileno())
            except OSError:
                # Sometimes fails on non-standard streams, safe to ignore
                pass

    def close(self):
        """
        Closes the file handle properly.
        """
        if self.log_file:
            self.log_file.close()

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global Exception Hook.
    This function is called automatically by Python whenever an unhandled exception occurs.
    It writes the full traceback to the log file, which is crucial for debugging GUI crashes.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to stop the program normally without logging it as a crash
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Format the error message header
    error_header = "\n" + "!"*60 + "\n"
    error_title = "   CRITICAL SYSTEM ERROR (UNHANDLED EXCEPTION)   \n"
    error_footer = "!"*60 + "\n"
    
    # Get the string representation of the full traceback
    trace_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    full_error_msg = f"{error_header}{error_title}{error_footer}\n{trace_str}\n"
    
    # Print to console (using the original stdout just in case our logger is broken)
    print(full_error_msg)
    
    # Write to log file if the logger is active
    if _logger_instance and _logger_instance.log_file:
        _logger_instance.log_file.write(full_error_msg)
        _logger_instance.flush()

def setup_logging(filename="debug_output.txt"):
    """
    Sets up the redirection of stdout and stderr.
    Must be called at the very start of the application entry point.
    """
    global _logger_instance
    
    # Create the dual logger instance
    _logger_instance = DualLogger(filename)
    
    # Redirect standard output (print statements)
    sys.stdout = _logger_instance
    
    # Redirect standard error (tracebacks and warnings)
    sys.stderr = _logger_instance
    
    # Hook into the global exception handler
    sys.excepthook = handle_exception
    
    print("[SYSTEM] Debug Logger Initialized.")
    print(f"[SYSTEM] Log file location: {os.path.abspath(filename)}")
    print("[SYSTEM] Global Exception Hook is active.")