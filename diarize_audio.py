#!/usr/bin/env python3
# NeMo diarization script for audio files

import os
import sys
import glob
import torch
import json
import argparse
import tempfile
from omegaconf import OmegaConf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NeMo Speaker Diarization")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="input", 
        help="Cesta ke složce s audio soubory (výchozí: 'input')"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Cesta k výstupní složce (výchozí: automaticky vytvořeno vedle vstupní složky)"
    )
    return parser.parse_args()

def create_manifest_file(audio_file, temp_dir):
    """
    Vytvoří dočasný manifest soubor pro NeMo diarizaci.
    """
    manifest_content = {
        'audio_filepath': audio_file,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': None,
        'rttm_filepath': None,
        'uem_filepath': None
    }
    
    # Uložení do JSON souboru
    manifest_filepath = os.path.join(temp_dir, 'manifest.json')
    with open(manifest_filepath, 'w') as f:
        json.dump(manifest_content, f)
    
    return manifest_filepath

def preprocess_audio(audio_file, temp_dir):
    """
    Preprocess audio file to ensure compatibility with NeMo diarization.
    """
    try:
        import soundfile as sf
        import librosa
        import numpy as np
        
        print(f"  Preprocessing audio file: {os.path.basename(audio_file)}")
        
        # Load audio file with librosa
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Ensure the audio is in the correct format (16kHz, mono)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Save as WAV file with soundfile
        base_filename = os.path.splitext(os.path.basename(audio_file))[0]
        temp_wav_path = os.path.join(temp_dir, f"{base_filename}_processed.wav")
        sf.write(temp_wav_path, y, sr, subtype='PCM_16')
        
        print(f"  Audio preprocessed and saved to: {temp_wav_path}")
        return temp_wav_path
    
    except Exception as e:
        print(f"  Error preprocessing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return audio_file  # Return original file if preprocessing fails

def main():
    # Parse command line arguments
    args = parse_args()
    input_dir = args.input_dir
    
    print("NeMo Speaker Diarization")
    print("========================")
    print(f"Vstupní složka: {input_dir}")
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"CHYBA: Vstupní složka '{input_dir}' neexistuje.")
        print(f"Vytvořit složku? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            try:
                os.makedirs(input_dir, exist_ok=True)
                print(f"Složka '{input_dir}' byla vytvořena.")
            except Exception as e:
                print(f"Nepodařilo se vytvořit složku: {str(e)}")
                return
        else:
            return
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(input_dir), "output")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Výstupní složka: {output_dir}")
    
    # Create a temporary directory for manifest files
    temp_dir = tempfile.mkdtemp()
    print(f"Dočasné soubory: {temp_dir}")
    
    # Find audio files in input folder
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not audio_files:
        print(f"Nenalezeny žádné audio soubory ve složce '{input_dir}'.")
        print(f"Prosím, umístěte audio soubory do složky '{input_dir}' a spusťte skript znovu.")
        return
    
    print(f"Nalezeno {len(audio_files)} audio souborů:")
    for i, audio_file in enumerate(audio_files):
        print(f"  {i+1}. {os.path.basename(audio_file)}")
    
    try:
        # Import NeMo diarization module
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        
        # Zpracování každého audio souboru
        for i, audio_file in enumerate(audio_files):
            print(f"\nZpracování souboru {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            try:
                # Preprocess audio file
                preprocessed_audio_file = preprocess_audio(audio_file, temp_dir)
                
                # Vytvoření dočasného manifestu pro tento soubor
                manifest_filepath = create_manifest_file(preprocessed_audio_file, temp_dir)
                
                print("\nInicializace diarizačního modelu...")
                
                # Detekce dostupného zařízení
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Použití zařízení: {device}")
                
                # Komplexní struktura konfigurace diarizéru podle poskytnutého konfiguračního slovníku
                nemo_diar_config = {
                    'device': device,
                    'batch_size': 1,  # Added batch_size parameter
                    'num_workers': 0,  # Added num_workers parameter
                    'sample_rate': 16000,  # Added sample_rate parameter
                    'verbose': True,  # Added verbose parameter
                    'diarizer': {
                        'manifest_filepath': manifest_filepath,
                        'out_dir': output_dir,
                        'paths2audio_files': [preprocessed_audio_file],
                        'num_speakers': None,
                        'oracle_vad': False,
                        'collar': 0.25,
                        'ignore_overlap': False,
                        'vad': {
                            'model_path': 'vad_multilingual_marblenet',
                            'parameters': {
                                'window_length_in_sec': 0.15,
                                'shift_length_in_sec': 0.01,
                                'overlap': 0.5,
                                'smoothing': 'median',
                                'window_size_for_smoothing': 3,
                                'threshold': 0.5,
                                'onset': 0.5,
                                'offset': 0.5,
                            }
                        },
                        'msdd_model': {
                            'model_path': 'diar_msdd_telephonic',
                            'parameters': {
                                'use_speaker_model_from_ckpt': True,
                                'diar_window_length': 50,
                                'diar_shift_length': 10,
                                'speaker_overlap_threshold': 0.5,
                                'infer_batch_size': 1,
                                'seq_eval_mode': True,
                                'diar_eval_settings': [(0.25, True), (0.25, False), (0.0, False)],
                                'max_num_of_spks': 20,
                                'overlap_infer_spk_limit': 10
                            }
                        },
                        'speaker_embeddings': {
                            'model_path': 'titanet_large',
                            'parameters': {
                                'window_length_in_sec': [1.5],  
                                'shift_length_in_sec': [0.75],  
                                'embed_batch_size': 64,
                                'device': 'cuda',
                                'multiscale_weights': [1.0],
                                'save_embeddings': False
                            }
                        },
                        'clustering': {
                            'method': 'cosine',
                            'parameters': {
                                'max_num_speakers': 20,
                                'sparse_search_volume': 30,
                                'enhanced_count_thres': 80,
                                'pval_thres': 0.02,
                                'max_rp_threshold': 0.15,
                                'oracle_num_speakers': False
                            }
                        }
                    },
                    'vad': {
                        'model_path': 'vad_multilingual_marblenet',
                        'parameters': {
                            'window_length_in_sec': 0.15,
                            'shift_length_in_sec': 0.01,
                            'smoothing': 'median',
                            'window_size_for_smoothing': 31,
                            'threshold': 0.4,
                            'onset': 0.4,
                            'offset': 0.3,
                            'pad_onset': 0.1,
                            'pad_offset': 0.2
                        }
                    },
                    'asr': {
                        'model_path': 'stt_en_conformer_ctc_large',
                        'parameters': {
                            'asr_based_vad': True,
                            'word_ts_anchor_offset': 0.15,
                            'ctc_decoder_params': {
                                'beam_size': 16,
                                'hotwords': []
                            }
                        }
                    },
                    'parallel': {
                        'vad_parallel_workers': 4,
                        'embedding_parallel_batches': 8,
                        'cluster_parallel_threads': 2
                    },
                    'device_settings': {
                        'vad_device': 'cpu',
                        'embedding_device': 'cuda:0',
                        'force_cpu': False
                    },
                    'preprocessing': {
                        'resample': True,
                        'target_sample_rate': 16000,
                        'audio_channel': 'mix'
                    },
                    'validation': {
                        'check_audio_length': True,
                        'min_audio_length': 0.5,
                        'max_audio_length': 36000,
                        'sample_rate': 16000
                    }
                }
                
                # Inicializace diarizéru s komplexní konfigurací
                diarizer_config = OmegaConf.create(nemo_diar_config)
                diarizer = NeuralDiarizer(cfg=diarizer_config)
                
                print("Model úspěšně inicializován!")
                
                # Spuštění diarizace
                print(f"  Probíhá diarizace...")
                diarizer.diarize()
                
                # Po diarizaci
                base_filename = os.path.splitext(os.path.basename(audio_file))[0]
                rttm_file = os.path.join(output_dir, f"{base_filename}.rttm")
                
                print(f"  Diarizace dokončena!")
                print(f"  Výsledky uloženy v souboru: {rttm_file}")
                
                # Zobrazení obsahu RTTM souboru
                if os.path.exists(rttm_file):
                    print("\n  Výsledky diarizace (prvních 5 řádků):")
                    with open(rttm_file, 'r') as f:
                        lines = f.readlines()
                        for line_idx, line in enumerate(lines[:5]):
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                start_time = float(parts[3])
                                duration = float(parts[4])
                                speaker = parts[7]
                                print(f"    Mluvčí {speaker}: {start_time:.2f}s - {start_time+duration:.2f}s (trvání: {duration:.2f}s)")
                        
                        if len(lines) > 5:
                            print(f"    ... a dalších {len(lines)-5} segmentů")
                else:
                    print("  VAROVÁNÍ: RTTM soubor nebyl vytvořen!")
                    
            except Exception as e:
                print(f"  CHYBA při zpracování souboru {os.path.basename(audio_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\nZpracování dokončeno!")
        print(f"Výsledky diarizace jsou uloženy v RTTM souborech ve složce: {output_dir}")
        
    except Exception as e:
        print(f"\nChyba: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPoužití NeMo pro diarizaci selhalo. Ujistěte se, že máte správně nainstalované všechny závislosti:")
        print("pip install nemo_toolkit['asr'] omegaconf")
    
    finally:
        # Odstranění dočasných souborů
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Dočasné soubory byly odstraněny.")
        except:
            print(f"Nepodařilo se odstranit dočasné soubory v {temp_dir}")

if __name__ == "__main__":
    main()
