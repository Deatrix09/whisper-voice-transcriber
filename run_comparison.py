import argparse
from src.utils.config import AppConfig
from src.utils.path_utils import PathUtils
from algorithm.levenshtein_comparator import LevenshteinComparator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare transcription results using Levenshtein distance")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = AppConfig.from_yaml(args.config)
    
    # Validate input path
    input_path, _ = PathUtils.validate_and_prepare_paths(args.input)
    
    # Initialize comparator
    comparator = LevenshteinComparator(config)
    
    # Process audio and get comparison results
    results = comparator.process_audio(input_path)
    
    # Get transcriptions from the process_audio method
    transcriptions = results.pop('transcriptions', {})  # Remove transcriptions from results dict
    
    # Save results
    comparator.save_results(results, input_path, transcriptions)

if __name__ == "__main__":
    main()