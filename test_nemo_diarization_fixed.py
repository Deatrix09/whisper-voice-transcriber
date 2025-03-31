#!/usr/bin/env python3
# NeMo speaker diarization test script

import os
import sys

# Create input directory if it doesn't exist
os.makedirs('input', exist_ok=True)

def main():
    print("Testing NeMo Speaker Diarization capability...")
    print(f"Python version: {sys.version}")
    
    # Import torch inside the function to avoid scope issues
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Import NeMo
        import nemo
        print(f"\nNeMo version: {nemo.__version__}")
        
        # Check if we can import speaker diarization modules
        print("\nChecking for diarization modules in NeMo...")
        
        # Try to import speaker embeddings model
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            print("✓ Successfully imported EncDecSpeakerLabelModel")
            
            # List available speaker models
            print("\nAvailable speaker embedding models:")
            models = EncDecSpeakerLabelModel.list_available_models()
            for idx, model_info in enumerate(models):
                print(f"  - {model_info.pretrained_model_name}: {model_info.description[:100]}...")
        except ImportError as e:
            print(f"✗ Could not import EncDecSpeakerLabelModel: {str(e)}")
        
        # Try to import diarization modules
        try:
            from nemo.collections.asr.models import NeuralDiarizer
            print("✓ Successfully imported NeuralDiarizer")
            
            # Try to list available diarization models
            try:
                print("\nAvailable diarization models:")
                diar_models = NeuralDiarizer.list_available_models()
                for idx, model_info in enumerate(diar_models):
                    print(f"  - {model_info.pretrained_model_name}")
                    
                if not diar_models:
                    print("  No pre-trained diarization models found in the registry")
            except Exception as e:
                print(f"  Error listing diarization models: {str(e)}")
        except ImportError as e:
            print(f"✗ Could not import NeuralDiarizer: {str(e)}")
            
            # Try to import MSDD models (used for diarization)
            try:
                from nemo.collections.asr.models.msdd_models import NeuralDiarizer
                print("✓ Successfully imported NeuralDiarizer from msdd_models")
            except ImportError as e:
                print(f"✗ Could not import NeuralDiarizer from msdd_models: {str(e)}")
        
        # Print instructions for using diarization in a real script
        print("\nExample code for using NeMo diarization on audio files in input folder:")
        print("""
# Sample code for diarization (modify paths as needed):
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

# 1. Configuration for diarizer
diarizer_config = {
    'speaker_embeddings': {
        'model_path': 'titanet_large',
        'parameters': {
            'window_length_in_sec': 1.5,
            'shift_length_in_sec': 0.75
        }
    },
    'clustering': {
        'parameters': {
            'oracle_num_speakers': False
        }
    },
    'msdd_model': {
        'model_path': 'diar_msdd_telephonic'
    }
}

# 2. Initialize diarizer
diarizer_config = OmegaConf.create(diarizer_config)
diarizer = NeuralDiarizer(cfg=diarizer_config)

# 3. Run diarization on an audio file
diarizer.diarize("input/your_audio_file.wav")
# This will generate an RTTM file with speaker segments
        """)
        
        # Check for missing dependencies
        print("\nChecking for common dependencies needed for diarization:")
        missing_deps = []
        
        try:
            import scipy
            print("✓ scipy is installed")
        except ImportError:
            missing_deps.append("scipy")
            print("✗ scipy is not installed")
            
        try:
            import torch.fft
            print("✓ torch.fft is available")
        except ImportError:
            missing_deps.append("torch with fft support")
            print("✗ torch.fft is not available")
            
        try:
            import torchaudio
            print("✓ torchaudio is installed")
        except ImportError:
            missing_deps.append("torchaudio")
            print("✗ torchaudio is not installed")
            
        try:
            import omegaconf
            print("✓ omegaconf is installed")
        except ImportError:
            missing_deps.append("omegaconf")
            print("✗ omegaconf is not installed")
            
        if missing_deps:
            print("\nMissing dependencies for diarization:")
            for dep in missing_deps:
                print(f"  - {dep}")
            print("\nInstall missing dependencies with:")
            print("pip install scipy torchaudio omegaconf")
            
        print("\nTo use diarization with NeMo, place your audio files in the input/ folder")
        print("and run the script with the example code shown above.")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("NeMo might not be installed correctly or there might be issues with dependencies.")

if __name__ == "__main__":
    main()
