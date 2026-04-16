# -*- coding: utf-8 -*-
import os
import socket
import time


def sanitize_file_component(value: str) -> str:
    value = str(value).strip()
    value = os.path.basename(value)
    value = os.path.splitext(value)[0]
    for char in (" ", "/", "\\", ":", ";"):
        value = value.replace(char, "_")
    return value or "unknown"


def get_timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def build_output_file_stem(model_name: str, dataset_name: str) -> str:
    return "_".join(
        [
            sanitize_file_component(model_name),
            sanitize_file_component(dataset_name),
            get_timestamp_string(),
        ]
    )


def get_unique_file_suffix():
    """
    Generate a log file name suffix that includes the following information:

    - Hostname
    - The current timestamp, in seconds, is the number of seconds since the Unix era
    - PID (process identifier) of the process

    Return:
    str: The name of the generated log file, in the format '.timestamp.hostname.pid.csv'

    For example, if the host name is' myhost ', the current timestamp is 1631655702, and the current process ID is 12345
    The returned file name may be '.1631655702.myhost.12345.csv'.
    """
    # Get Host Name
    hostname = socket.gethostname()

    # Get current timestamp (seconds since Unix era)
    timestamp = int(time.time())

    # Obtain the PID (process identifier) of the process
    pid = os.getpid()

    # Build file name
    log_filename = f".{timestamp}.{hostname}.{pid}.csv"
    return log_filename
