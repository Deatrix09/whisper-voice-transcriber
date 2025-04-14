#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run main.py with multiple configuration files from a specified folder"
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        required=True,
        help="Path to folder containing configuration files (subfolder within config/)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input audio file to process with all configurations"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "txt", "subtitle"],
        help="Output format (optional, will use default from config if not specified)"
    )
    parser.add_argument(
        "--python",
        type=str,
        help="Path to specific Python executable to use (optional, will use system Python if not specified)"
    )
    
    return parser.parse_args()


def get_config_files(config_folder):
    """Get list of all YAML files in the specified folder."""
    if not os.path.exists(config_folder):
        print(f"Error: Configuration folder '{config_folder}' not found.")
        sys.exit(1)
        
    config_files = []
    for file in os.listdir(config_folder):
        if file.endswith(".yaml") or file.endswith(".yml"):
            config_files.append(os.path.join(config_folder, file))
    
    if not config_files:
        print(f"Error: No YAML configuration files found in '{config_folder}'.")
        sys.exit(1)
        
    return sorted(config_files)


def get_python_executable():
    """
    Determine the correct Python executable to use.
    This will try to use the current virtual environment's Python if available.
    """
    # Check if running in a virtual environment
    if sys.prefix != sys.base_prefix:
        # We're in a virtual environment, use its Python
        if os.name == 'nt':  # Windows
            return os.path.join(sys.prefix, 'Scripts', 'python.exe')
        else:  # Linux, macOS
            return os.path.join(sys.prefix, 'bin', 'python')
    
    # Check for Conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        if os.name == 'nt':  # Windows
            return os.path.join(conda_prefix, 'python.exe')
        else:  # Linux, macOS
            return os.path.join(conda_prefix, 'bin', 'python')
    
    # Otherwise, use the current Python interpreter
    return sys.executable


def main():
    """Main function to process multiple configurations."""
    args = parse_arguments()
    
    # Determine which Python executable to use
    python_executable = args.python if args.python else get_python_executable()
    print(f"Using Python executable: {python_executable}")
    
    # Convert config folder path to absolute if it's relative
    if not os.path.isabs(args.config_folder):
        if args.config_folder.startswith("config/") or args.config_folder.startswith("config\\"):
            # Already relative to project root
            config_folder = args.config_folder
        else:
            # Prepend config/ if not already there
            config_folder = os.path.join("config", args.config_folder)
    else:
        config_folder = args.config_folder
    
    # Get all configuration files
    config_files = get_config_files(config_folder)
    total_configs = len(config_files)
    
    print(f"Found {total_configs} configuration files in '{config_folder}'")
    print(f"Input audio file: {args.input}")
    
    # Process each configuration file
    start_time = time.time()
    success_count = 0
    failure_count = 0
    
    print("\n" + "="*60)
    print(f"STARTING BATCH PROCESSING AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    for idx, config_file in enumerate(config_files, 1):
        config_name = os.path.basename(config_file)
        print(f"\n[{idx}/{total_configs}] Processing with configuration: {config_name}")
        
        # Build command
        cmd = [python_executable, "main.py", "--config", config_file, "--input", args.input]
        if args.format:
            cmd.extend(["--format", args.format])
            
        # Run main.py with current configuration
        print(f"Running: {' '.join(cmd)}")
        print("-"*60)
        
        try:
            start_config_time = time.time()
            process = subprocess.run(cmd, check=True, text=True, capture_output=True)
            config_duration = time.time() - start_config_time
            
            # Print stdout from process
            print(process.stdout)
            
            success_count += 1
            print(f"Completed successfully in {config_duration:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            failure_count += 1
            print(f"Failed with exit code {e.returncode}")
            print("Error output:")
            print(e.stdout)
            print(e.stderr)
            
        except Exception as e:
            failure_count += 1
            print(f"Error occurred: {str(e)}")
            
        print("-"*60)
        print(f"Progress: {idx}/{total_configs} configurations processed")
    
    # Print summary
    total_duration = time.time() - start_time
    print("\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Total configs processed: {total_configs}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    


# python run_multiple_configs.py --config-folder config/fine_tuning --input path/to/your/audio/file.mp3 --format txt

# Make sure to run this script from the project root directory where main.py is located.
if __name__ == "__main__":
    main()