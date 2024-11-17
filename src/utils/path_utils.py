import os
from pathlib import Path
import logging
from datetime import datetime

class PathUtils:
    @staticmethod
    def get_default_output_path(input_path: str, output_format: str = "txt") -> str:
        """
        Generate default output path based on input file and format.
        Returns path like: data/output/filename_YYYYMMDD_HHMMSS.{format}
        """
        # Get input filename without extension
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_filename = f"{input_filename}_{timestamp}.{output_format}"
        
        output_path = os.path.join(os.getcwd(), 'data', 'output', output_filename)
        
        return output_path

    @staticmethod
    def validate_and_prepare_paths(input_path: str, output_path: str = None, output_format: str = None) -> tuple[str, str]:
        """
        Validate input path and prepare output path.
        Returns absolute paths for both input and output.
        """
        logger = logging.getLogger(__name__)
        
        # Set default format to txt if not specified
        if output_format is None:
            output_format = "txt"
            logger.info("No output format specified, using default: txt")
        
        # Convert input path to absolute path
        abs_input_path = os.path.abspath(input_path)
        
        # Check if input file exists
        if not os.path.exists(abs_input_path):
            raise FileNotFoundError(f"Input audio file not found: {abs_input_path}")
        
        # Generate default output path if none provided
        if output_path is None:
            output_path = PathUtils.get_default_output_path(input_path, output_format)
            logger.info(f"Using default output path: {output_path}")
            
        abs_output_path = os.path.abspath(output_path)
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(abs_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
            
        # Create data directory structure if it doesn't exist
        data_dir = os.path.join(os.getcwd(), 'data')
        input_dir = os.path.join(data_dir, 'input')
        output_dir = os.path.join(data_dir, 'output')
        
        for directory in [data_dir, input_dir, output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        
        return abs_input_path, abs_output_path

    @staticmethod
    def ensure_directory_exists(path: str) -> str:
        """Ensure directory exists and create if it doesn't."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)