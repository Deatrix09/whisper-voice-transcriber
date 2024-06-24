import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)