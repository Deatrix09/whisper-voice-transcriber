import os
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from src.utils.logger import setup_logger
from src.core.pipeline import Pipeline
from src.utils.config import AppConfig
import numpy as np

class LevenshteinComparator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file)
        self.models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing extra whitespace and converting to lowercase.
        """
        return ' '.join(text.lower().split())

    def get_transcription_text(self, results: List[Dict]) -> str:
        """
        Extract and concatenate all text from transcription results.
        """
        return ' '.join([segment['text'].strip() for segment in results])

    def calculate_similarity_matrix(self, transcriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate similarity matrix between all model transcriptions.
        Returns both raw distances and normalized similarity scores.
        """
        models = list(transcriptions.keys())
        n_models = len(models)
        
        # Initialize matrices
        distances = np.zeros((n_models, n_models))
        similarities = np.zeros((n_models, n_models))
        
        # Calculate distances and similarities
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i <= j:  # Only calculate upper triangle
                    text1 = transcriptions[model1]
                    text2 = transcriptions[model2]
                    
                    # Calculate Levenshtein distance
                    distance = self.levenshtein_distance(text1, text2)
                    distances[i, j] = distance
                    distances[j, i] = distance  # Mirror the matrix
                    
                    # Calculate normalized similarity score (0-100%)
                    max_len = max(len(text1), len(text2))
                    similarity = round(((max_len - distance) / max_len) * 100, 3)
                    similarities[i, j] = similarity
                    similarities[j, i] = similarity  # Mirror the matrix
        
        # Create DataFrames
        distance_df = pd.DataFrame(distances, index=models, columns=models)
        similarity_df = pd.DataFrame(similarities, index=models, columns=models)
        
        return distance_df, similarity_df

    def process_audio(self, audio_path: str) -> Dict[str, pd.DataFrame]:
        """
        Process audio file with different models and compare results.
        """
        try:
            self.logger.info(f"Starting Levenshtein comparison for: {audio_path}")
            
            # Store transcriptions for each model
            transcriptions = {}
            
            # Process with each model
            for model in self.models:
                self.logger.info(f"Processing with model: {model}")
                
                # Update config for current model
                self.config.model.name = model
                
                # Initialize pipeline with current model
                pipeline = Pipeline(self.config)
                
                # Process audio
                results = pipeline.process_file(audio_path)
                
                # Extract and normalize text
                text = self.normalize_text(self.get_transcription_text(results))
                transcriptions[model] = text
                
                self.logger.info(f"Completed processing with {model}")
            
            # Calculate similarity matrices
            distance_matrix, similarity_matrix = self.calculate_similarity_matrix(transcriptions)
            
            # Prepare results
            results = {
                'levenshtein_distances': distance_matrix,
                'similarity_scores': similarity_matrix,
                'transcriptions': transcriptions  # Include transcriptions in the results
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Levenshtein comparison: {str(e)}")
            raise

    def save_results(self, results: Dict[str, pd.DataFrame], audio_path: str, transcriptions: Dict[str, str]) -> None:
        """
        Save comparison results to files.
        """
        try:
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(os.getcwd(), 'data', 'algorithm_output', audio_name, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save matrices to CSV
            for name, df in results.items():
                output_path = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(output_path)
                self.logger.info(f"Saved {name} to: {output_path}")
            
            # Generate summary report
            self.generate_summary_report(results, output_dir)
            
            # Add visualization of differences
            self.visualize_differences(transcriptions, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def generate_summary_report(self, results: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        Generate a summary report of the comparison results.
        """
        try:
            similarity_scores = results['similarity_scores']
            
            # Calculate average similarity for each model
            avg_similarities = similarity_scores.mean()
            
            # Find most similar and most different model pairs (excluding self-comparisons)
            np.fill_diagonal(similarity_scores.values, -np.inf)  # Exclude diagonal (self-comparisons)
            max_similarity = similarity_scores.max().max()
            
            np.fill_diagonal(similarity_scores.values, np.inf)  # Set diagonal to inf for finding min
            min_similarity = similarity_scores.min().max()  # Find minimum similarity
            
            # Get model pairs for max/min similarities
            max_pair = np.where(results['similarity_scores'] == max_similarity)
            min_pair = np.where(results['similarity_scores'] == min_similarity)
            
            # Create summary report
            summary = [
                "Shrnutí srovnání modelů",
                "=====================",
                "",
                "Průměrné skóre podobností:",
                "----------------------"
            ]
            
            for model, score in avg_similarities.items():
                summary.append(f"{model}: {score:.2f}%")
            
            summary.extend([
                "",
                "Nejvíc podobný model:",
                f"{similarity_scores.index[max_pair[0][0]]} - {similarity_scores.columns[max_pair[1][0]]}",
                f"Podobnost: {max_similarity:.2f}%",
                "",
                "Nejvíc odlišný model:",
                f"{similarity_scores.index[min_pair[0][0]]} - {similarity_scores.columns[min_pair[1][0]]}",
                f"Podobnost: {min_similarity:.2f}%"
            ])
            
            # Save summary report
            summary_path = os.path.join(output_dir, "summary_report.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary))
            
            self.logger.info(f"Generated summary report: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise 

    def visualize_differences(self, transcriptions: Dict[str, str], output_dir: str) -> None:
        """
        Vytvoří přehledné porovnání textů s vizuálním zvýrazněním rozdílů.
        """
        try:
            # Připravíme výstupní soubor
            output_path = os.path.join(output_dir, "text_comparisons.txt")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Porovnání transkripcí mezi modely\n")
                f.write("================================\n\n")
                
                # Pro každý model vypíšeme text a základní statistiky
                for model, text in transcriptions.items():
                    f.write(f"\n{model.upper()}\n")
                    f.write("-" * len(model) + "\n")
                    f.write(f"Délka textu: {len(text)} znaků\n")
                    f.write(f"Text: {text}\n\n")
                
                # Porovnání mezi všemi páry modelů
                f.write("\nPorovnání rozdílů mezi modely\n")
                f.write("============================\n\n")
                
                models = list(transcriptions.keys())
                for i, model1 in enumerate(models):
                    for model2 in models[i+1:]:
                        text1 = transcriptions[model1]
                        text2 = transcriptions[model2]
                        distance = self.levenshtein_distance(text1, text2)
                        similarity = round(((max(len(text1), len(text2)) - distance) / max(len(text1), len(text2))) * 100, 3)
                        
                        f.write(f"{model1} vs {model2}\n")
                        f.write(f"Podobnost: {similarity:.2f}%\n")
                        f.write(f"Levenshteinova vzdálenost: {distance}\n\n")
            
            self.logger.info(f"Uloženo porovnání textů do: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Chyba při vytváření vizualizace rozdílů: {str(e)}")
            raise