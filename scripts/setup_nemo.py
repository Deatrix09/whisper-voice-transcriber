#!/usr/bin/env python3
"""
Setup script for NVIDIA NeMo toolkit for speaker diarization.

This script installs the NeMo toolkit and downloads the required models
for speaker diarization.
"""

import argparse
import os
import platform
import subprocess
import sys
import venv
import pkg_resources
import shutil
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Set up NVIDIA NeMo toolkit for speaker diarization"
    )
    parser.add_argument(
        "--venv", type=str, default=None,
        help="Create and use a virtual environment with the specified name"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force reinstallation even if NeMo is already installed"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Install with CUDA support (recommended for GPU users)"
    )
    return parser.parse_args()

def check_python_version():
    """Check if Python version is compatible with NeMo (>=3.8)"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"Error: Python 3.8 or higher is required, but you have {platform.python_version()}")
        sys.exit(1)
    print(f"Python version {platform.python_version()} is compatible with NeMo")

def create_virtual_environment(venv_path):
    """Create a virtual environment at the specified path"""
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    
    # Determine the Python executable path in the virtual environment
    if platform.system() == "Windows":
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_exe = os.path.join(venv_path, "bin", "python")
        pip_exe = os.path.join(venv_path, "bin", "pip")
    
    # Update pip in the virtual environment
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    
    return python_exe, pip_exe

def check_pytorch_installation(python_exe, cuda_support=False):
    """Check if PyTorch is installed correctly"""
    try:
        if cuda_support:
            print("Installing PyTorch with CUDA support...")
            subprocess.check_call([
                python_exe, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            print("Installing PyTorch (CPU version)...")
            subprocess.check_call([
                python_exe, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ])

        # Verify PyTorch installation
        result = subprocess.run(
            [python_exe, "-c", "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"],
            capture_output=True,
            text=True
        )
        print(result.stdout.strip())
        
        if "Import Error" in result.stderr:
            print("Warning: PyTorch installation may have issues")
            print(result.stderr)
            
        return True
    
    except subprocess.CalledProcessError:
        print("Failed to install PyTorch")
        return False

def install_nemo(python_exe, pip_exe, force=False):
    """Install NeMo toolkit"""
    try:
        # Check if NeMo is already installed
        if not force:
            try:
                subprocess.run(
                    [python_exe, "-c", "import nemo"],
                    check=True,
                    capture_output=True
                )
                print("NeMo is already installed. Use --force to reinstall.")
                return True
            except subprocess.CalledProcessError:
                pass  # Not installed, continue with installation
        
        print("Installing NeMo toolkit...")
        
        # Install dependencies first
        subprocess.check_call([
            pip_exe, "install", "Cython", "numpy", "scipy"
        ])
        
        # Try to install from PyPI first
        try:
            subprocess.check_call([
                pip_exe, "install", "nemo_toolkit>=1.22.0"
            ])
        except subprocess.CalledProcessError:
            print("PyPI installation failed, trying from GitHub...")
            subprocess.check_call([
                pip_exe, "install", "git+https://github.com/NVIDIA/NeMo.git"
            ])
        
        # Additional dependencies
        subprocess.check_call([
            pip_exe, "install", "soundfile", "gradio>=3.0.0"
        ])
        
        # Verify installation
        subprocess.check_call([
            python_exe, "-c", "import nemo; print(f'NeMo version: {nemo.__version__}')"
        ])
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing NeMo: {e}")
        return False

def download_nemo_models(python_exe):
    """Download pretrained NeMo models for diarization"""
    print("Downloading NeMo models for speaker diarization...")
    
    # Download models script
    download_script = """
import nemo.collections.asr as nemo_asr

# Download VAD model
print("Downloading Voice Activity Detection model...")
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")
print(f"VAD model downloaded to {vad_model.model_path}")

# Download speaker embedding model
print("Downloading Speaker Embedding model...")
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
print(f"Speaker model downloaded to {speaker_model.model_path}")

# Optional: Download MSDD model
try:
    print("Downloading Multiscale Diarization Decoder model...")
    msdd_model = nemo_asr.models.ClusteringDiarizer.from_pretrained(model_name="diar_msdd_telephonic")
    print(f"MSDD model downloaded to {msdd_model.model_path}")
except Exception as e:
    print(f"MSDD model download failed: {e}")
    print("This is optional, you can still use NeMo for diarization without it.")
"""
    
    try:
        with open("download_models.py", "w") as f:
            f.write(download_script)
            
        subprocess.check_call([python_exe, "download_models.py"])
        os.remove("download_models.py")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error downloading models: {e}")
        return False

def update_config_file():
    """Update the config file with NeMo settings"""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("Config file not found")
        return False
    
    try:
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Check if NeMo configuration already exists
        if "nemo:" in config_content:
            print("NeMo configuration already exists in config file")
            return True
            
        # Find the diarization section in the config
        diarization_indent = None
        lines = config_content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("diarization:"):
                diarization_indent = len(line) - len(line.lstrip())
                break
        
        if diarization_indent is None:
            print("Could not find diarization section in config file")
            return False
        
        # Add NeMo configuration
        indent = " " * (diarization_indent + 2)
        nemo_config = f"\n{indent}module: \"pyannote\"  # Choose between \"pyannote\" or \"nemo\"\n"
        
        # Add NeMo section
        nemo_config += f"\n{indent}nemo:\n"
        nemo_config += f"{indent}  enabled: false\n"
        nemo_config += f"{indent}  vad_model: \"vad_multilingual_marblenet\"\n"
        nemo_config += f"{indent}  speaker_model: \"titanet_large\"\n"
        nemo_config += f"{indent}  msdd_model: null\n"
        nemo_config += f"{indent}  clustering_params:\n"
        nemo_config += f"{indent}    method: \"sc\"\n"
        nemo_config += f"{indent}    min_samples: 2\n"
        nemo_config += f"{indent}    oracle_num_speakers: false\n"
        nemo_config += f"{indent}    max_num_speakers: 8\n"
        nemo_config += f"{indent}    enhanced_count_thres: 80\n"
        nemo_config += f"{indent}  vad_params:\n"
        nemo_config += f"{indent}    threshold: 0.5\n"
        nemo_config += f"{indent}    window_length_in_sec: 0.15\n"
        nemo_config += f"{indent}    shift_length_in_sec: 0.01\n"
        
        # Modify configuration file
        new_lines = []
        added = False
        for line in lines:
            new_lines.append(line)
            if line.startswith("diarization:"):
                # Add module parameter right after diarization:
                # Find the position to insert pyannote config
                for i, next_line in enumerate(lines[lines.index(line)+1:]):
                    if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= diarization_indent:
                        break
                    if "pyannote:" in next_line:
                        new_lines.append(nemo_config)
                        added = True
                        break
        
        if not added:
            # If we didn't find a place to add it, add it at the end of the diarization section
            new_lines.append(nemo_config)
        
        # Write updated content back to file
        with open(config_path, "w") as f:
            f.write("\n".join(new_lines))
            
        print("Updated config file with NeMo configuration")
        return True
    
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

def main():
    args = parse_arguments()
    
    print("Setting up NVIDIA NeMo toolkit for speaker diarization...")
    
    # Check Python version
    check_python_version()
    
    # Execute in virtual environment if specified
    python_exe = sys.executable
    pip_exe = "pip"
    
    if args.venv:
        # Create and activate virtual environment
        venv_path = os.path.abspath(args.venv)
        python_exe, pip_exe = create_virtual_environment(venv_path)
        
        print(f"Virtual environment created at: {venv_path}")
        if platform.system() == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"To activate the environment, run: {activate_cmd}")
    
    # Install/verify PyTorch
    if not check_pytorch_installation(python_exe, args.cuda):
        print("Failed to set up PyTorch. Exiting.")
        sys.exit(1)
    
    # Install NeMo
    if not install_nemo(python_exe, pip_exe, args.force):
        print("Failed to install NeMo. Exiting.")
        sys.exit(1)
    
    # Download models
    if not download_nemo_models(python_exe):
        print("Warning: Failed to download some NeMo models")
    
    # Update config file with NeMo settings
    update_config_file()
    
    print("\nNeMo setup completed successfully!")
    print("\nTo use NeMo for diarization, update your config.yaml file:")
    print("1. Set diarization.module to \"nemo\"")
    print("2. Set diarization.nemo.enabled to true")
    print("3. Optionally adjust the clustering and VAD parameters")
    
    if args.venv:
        print(f"\nRemember to activate your virtual environment with: {activate_cmd}")

if __name__ == "__main__":
    main()
