# Copyright (c) 2024-2026 Kleveta Lukas
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from src.utils.logger import setup_logger
from src.utils.config import AppConfig
import numpy as np

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback bez nové závislosti
    _RapidLevenshtein = None
    _HAS_RAPIDFUZZ = False

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:  # pragma: no cover - WER/CER se přeskočí s warningem
    jiwer = None
    _HAS_JIWER = False

_JIWER_TR = None


def _norm(text: str) -> str:
    """Normalizace pro WER/CER (fair napříč enginy: case + interpunkce pryč)."""
    if not _HAS_JIWER:
        return (text or "").lower()
    global _JIWER_TR
    if _JIWER_TR is None:
        _JIWER_TR = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip()])
    try:
        return _JIWER_TR(text or "")
    except Exception:
        return (text or "").lower()


def _python_levenshtein(s1: str, s2: str) -> int:
    """Referenční DP implementace (pomalá, ale bez závislostí)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current_row = [i]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _fast_levenshtein(s1: str, s2: str) -> int:
    if _HAS_RAPIDFUZZ:
        return int(_RapidLevenshtein.distance(s1, s2))
    return _python_levenshtein(s1, s2)


def _similarity_from_distance(text1: str, text2: str, distance: int) -> float:
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 100.0
    return round(((max_len - distance) / max_len) * 100, 3)

class LevenshteinComparator:
    # Defaultní sada (zpětná kompatibilita). --models filtr umí i turbo/parakeet.
    BASE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    EXTRA_MODELS = ["turbo", "parakeet"]
    KNOWN_MODELS = BASE_MODELS + EXTRA_MODELS
    # Reference pro WER/CER/cpWER = největší dostupný model v sadě.
    REFERENCE_PRIORITY = ["large-v3", "turbo", "large-v2", "parakeet",
                          "medium", "small", "base", "tiny"]

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(__name__, config.logging.level, config.logging.file,
                                     file_level=getattr(config.logging, "file_level", None),
                                     console_level=getattr(config.logging, "console_level", None))
        self.models = list(self.BASE_MODELS)

    def set_models(self, names) -> List[str]:
        """Nastav --models filtr (např. ["tiny", "turbo", "parakeet"])."""
        names = [str(n).strip() for n in (names or []) if str(n).strip()]
        if not names:
            raise ValueError(" --models je prázdný")
        unknown = [n for n in names if n not in self.KNOWN_MODELS]
        if unknown:
            raise ValueError(f"Neznámé modely: {unknown}. Možné: {self.KNOWN_MODELS}")
        self.models = names
        return self.models

    @staticmethod
    def _resolve_entry(name: str) -> Dict[str, str]:
        """Jméno -> engine. Parakeet ignoruje model.name (bere asr.parakeet_model)."""
        if name == "parakeet":
            return {"engine": "parakeet", "model_name": name}
        return {"engine": "faster-whisper", "model_name": name}

    @classmethod
    def _pick_reference(cls, transcriptions: Dict[str, str]) -> str:
        for cand in cls.REFERENCE_PRIORITY:
            if cand in transcriptions:
                return cand
        return next(iter(transcriptions))
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        return _fast_levenshtein(s1, s2)

    @staticmethod
    def similarity_from_distance(text1: str, text2: str, distance: int) -> float:
        return _similarity_from_distance(text1, text2, distance)

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
        models = list(transcriptions.keys())
        n_models = len(models)

        # Cache vzdáleností pro reuse ve visualize_differences (bez 2. přepočtu).
        # Klíče oběma směry, aby starší volání bez normalizace pořadí fungovala.
        self._last_distances: Dict[Tuple[str, str], int] = {}

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
                    self._last_distances[(model1, model2)] = int(distance)
                    self._last_distances[(model2, model1)] = int(distance)

                    # Calculate normalized similarity score (0-100%)
                    similarity = self.similarity_from_distance(text1, text2, int(distance))
                    similarities[i, j] = similarity
                    similarities[j, i] = similarity  # Mirror the matrix
        
        # Create DataFrames
        distance_df = pd.DataFrame(distances, index=models, columns=models)
        similarity_df = pd.DataFrame(similarities, index=models, columns=models)

        return distance_df, similarity_df

    def calculate_wer_cer(self, transcriptions: Dict[str, str],
                          reference: Optional[str] = None) -> "pd.DataFrame":
        """WER/CER každého modelu proti referenčnímu transcriptu.

        Reference default = největší model v sadě (_pick_reference).
        Sloupce: [model, wer, cer, reference]. Vyžaduje jiwer.
        """
        if not _HAS_JIWER:
            raise RuntimeError("jiwer není nainstalován. Spusť: pip install \"jiwer>=4.0\"")
        ref_key = reference or self._pick_reference(transcriptions)
        ref_text = _norm(transcriptions[ref_key])
        rows = []
        for model, text in transcriptions.items():
            if model == ref_key:
                wer, cer = 0.0, 0.0
            else:
                wer = float(jiwer.wer(ref_text, _norm(text)))
                cer = float(jiwer.cer(ref_text, _norm(text)))
            rows.append({"model": model, "wer": round(wer, 4), "cer": round(cer, 4),
                         "reference": ref_key})
        return pd.DataFrame(rows, columns=["model", "wer", "cer", "reference"])

    @staticmethod
    def _speaker_texts(segments: List[Dict]) -> Dict[str, str]:
        """{speaker: konkatenovaný text seřazený podle startu}."""
        groups: Dict[str, List[str]] = {}
        for s in sorted(segments, key=lambda x: x["start"]):
            groups.setdefault(s["speaker"], []).append(s["text"].strip())
        return {sp: " ".join(t) for sp, t in groups.items()}

    def _cpwer_pair(self, ref_groups: Dict[str, str],
                    hyp_groups: Dict[str, str]) -> float:
        """Concatenated minimum-permutation WER dvojice transcriptů.

        Permutace přiřazení hyp speakerů na ref (do 7+7, pak aproximace
        plain WER přes všechno). Prázdná reference: 0.0/1.0.
        """
        import itertools
        ref_sp = list(ref_groups)
        hyp_sp = list(hyp_groups)
        if not ref_sp:
            return 0.0 if not hyp_sp else 1.0
        if not hyp_sp:
            return 1.0
        if max(len(ref_sp), len(hyp_sp)) > 7:
            self.logger.warning("cpWER: >7 speakerů, aproximace plain WER")
            all_ref = _norm(" ".join(ref_groups[s] for s in ref_sp))
            all_hyp = _norm(" ".join(hyp_groups[s] for s in hyp_sp))
            return round(float(jiwer.wer(all_ref, all_hyp)), 4)
        k = max(len(ref_sp), len(hyp_sp))
        ref_list = [_norm(ref_groups[s]) for s in ref_sp] + [""] * (k - len(ref_sp))
        hyp_list = [_norm(hyp_groups[s]) for s in hyp_sp] + [""] * (k - len(hyp_sp))
        ref_concat = " ".join(ref_list)
        best = min(float(jiwer.wer(ref_concat, " ".join(p)))
                   for p in itertools.permutations(hyp_list))
        return round(best, 4)

    def calculate_cpwer(self, segments_per_model: Dict[str, List[Dict]],
                        reference: Optional[str] = None) -> Dict[str, float]:
        """cpWER každého modelu proti referenci (vyžaduje jiwer)."""
        if not _HAS_JIWER:
            raise RuntimeError("jiwer není nainstalován. Spusť: pip install \"jiwer>=4.0\"")
        ref_key = reference or self._pick_reference(
            {m: "" for m in segments_per_model})
        ref_groups = self._speaker_texts(segments_per_model[ref_key])
        return {m: (0.0 if m == ref_key else self._cpwer_pair(ref_groups, self._speaker_texts(s)))
                for m, s in segments_per_model.items()}

    def process_audio(self, audio_path: str) -> Dict[str, pd.DataFrame]:
        # Lazy import: těžký torch/whisper/pyannote řetězec až při reálném běhu,
        # aby unit testy a --help fungovaly i bez torchaudio.
        from src.core.pipeline import Pipeline
        import time as _time
        try:
            audio_dur = 0.0
            try:
                import soundfile as _sf
                audio_dur = float(_sf.info(audio_path).duration)
            except Exception:
                pass
            self.logger.info(f"Starting Levenshtein comparison for: {audio_path}")

            # Check if audio is M4A and convert once
            from src.audio.processor import AudioProcessor
            temp_processor = AudioProcessor(self.config)
            if audio_path.lower().endswith('.m4a'):
                self.logger.info(f"Converting M4A file once for all models")
                _, _, converted_path = temp_processor.load_audio(audio_path)
                if converted_path:
                    audio_path = converted_path
                    self.logger.info(f"Using converted WAV file: {audio_path}")

            # Store transcriptions for each model
            transcriptions = {}
            segments_per_model = {}
            timing_rows = []

            # Původní hodnoty configu pro restore (FÁZE 4+: asr.* má přednost).
            asr_cfg = getattr(self.config, "asr", None)
            orig_engine = getattr(asr_cfg, "engine", None)
            orig_asr_model = getattr(asr_cfg, "model", None)
            orig_name = self.config.model.name

            try:
                # Process with each model
                for model in self.models:
                    entry = self._resolve_entry(model)
                    self.logger.info(
                        f"Processing with model: {model} (engine={entry['engine']})")

                    # Update config for current model (oba klíče: FÁZE 4
                    # Transcriber čte asr.* s předností, bez toho by jel
                    # 6x stejný default model!)
                    if asr_cfg is not None:
                        asr_cfg.engine = entry["engine"]
                        if entry["engine"] == "faster-whisper":
                            asr_cfg.model = entry["model_name"]
                    self.config.model.name = entry["model_name"]

                    # Initialize pipeline with current model
                    pipeline = Pipeline(self.config)

                    t0 = _time.perf_counter()
                    try:
                        # Process audio
                        results = pipeline.process_file(audio_path)

                        # Extract and normalize text
                        text = self.normalize_text(self.get_transcription_text(results))
                        transcriptions[model] = text
                        segments_per_model[model] = results
                    finally:
                        # L03: uvolni Whisper/pyannote před dalším (větším) modelem,
                        # jinak OOM na 8GB VRAM a držení 6 modelů v GC.
                        Pipeline.release_model_resources(pipeline)
                        del pipeline

                    dt = _time.perf_counter() - t0
                    rtfx = (audio_dur / dt) if dt > 0 and audio_dur > 0 else 0.0
                    timing_rows.append({"model": model, "seconds": round(dt, 1),
                                        "audio_seconds": round(audio_dur, 1),
                                        "rtfx": round(rtfx, 2)})
                    self.logger.info(
                        f"Completed {model} in {dt:.1f}s (RTFx={rtfx:.2f})")
            finally:
                if asr_cfg is not None:
                    asr_cfg.engine = orig_engine
                    asr_cfg.model = orig_asr_model
                self.config.model.name = orig_name

            # Clean up temporary files
            temp_processor.cleanup()

            # Calculate similarity matrices
            distance_matrix, similarity_matrix = self.calculate_similarity_matrix(transcriptions)

            results = {
                'levenshtein_distances': distance_matrix,
                'similarity_scores': similarity_matrix,
                'timing': pd.DataFrame(timing_rows, columns=["model", "seconds",
                                                             "audio_seconds", "rtfx"]),
                'transcriptions': transcriptions
            }

            # WER/CER/cpWER vs reference (bez jiwer se přeskočí s warningem)
            if _HAS_JIWER and transcriptions:
                reference = self._pick_reference(transcriptions)
                wer_cer = self.calculate_wer_cer(transcriptions, reference)
                try:
                    cpwer = self.calculate_cpwer(segments_per_model, reference)
                    wer_cer["cpwer"] = wer_cer["model"].map(cpwer)
                except Exception as e:
                    self.logger.warning(f"cpWER selhalo ({e}), jen WER/CER")
                results['wer_cer'] = wer_cer
            elif transcriptions:
                self.logger.warning("jiwer chybí (pip install \"jiwer>=4.0\"), "
                                    "WER/CER/cpWER přeskočeny")

            return results

        except Exception as e:
            self.logger.error(f"Error in Levenshtein comparison: {str(e)}")
            raise

    def save_results(self, results: Dict[str, pd.DataFrame], audio_path: str, transcriptions: Dict[str, str]) -> None:
        try:
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(os.getcwd(), 'data', 'algorithm_output', audio_name, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            for name, df in results.items():
                output_path = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(output_path)
                self.logger.info(f"Saved {name} to: {output_path}")
            
            self.generate_summary_report(results, output_dir)

            self.visualize_differences(
                transcriptions, output_dir,
                distance_cache=getattr(self, "_last_distances", None),
            )
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def generate_summary_report(self, results: Dict[str, pd.DataFrame], output_dir: str) -> None:
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

            # FÁZE 6: rychlost (RTFx) + přesnost (WER/CER/cpWER vs reference)
            timing = results.get('timing')
            if timing is not None and len(timing):
                summary.extend(["", "Rychlost (RTFx = délka audia / čas):",
                                "----------------------"])
                for _, row in timing.iterrows():
                    summary.append(
                        f"{row['model']}: {row['rtfx']:.2f}x "
                        f"({row['seconds']:.1f}s / {row['audio_seconds']:.1f}s audia)")

            wer_cer = results.get('wer_cer')
            if wer_cer is not None and len(wer_cer):
                ref = wer_cer['reference'].iloc[0]
                summary.extend(["", f"Přesnost vs reference {ref} (WER/CER/cpWER):",
                                "----------------------"])
                for _, row in wer_cer.iterrows():
                    cp = row.get('cpwer', None)
                    cp_txt = f", cpWER {cp:.4f}" if cp is not None and pd.notna(cp) else ""
                    summary.append(f"{row['model']}: WER {row['wer']:.4f}, "
                                   f"CER {row['cer']:.4f}{cp_txt}")
            
            summary_path = os.path.join(output_dir, "summary_report.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary))
            
            self.logger.info(f"Generated summary report: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise 

    def visualize_differences(
        self,
        transcriptions: Dict[str, str],
        output_dir: str,
        distance_cache: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> None:
        try:
            output_path = os.path.join(output_dir, "text_comparisons.txt")
            cache = distance_cache or getattr(self, "_last_distances", None) or {}

            def _cached_distance(m1: str, m2: str, t1: str, t2: str) -> int:
                if (m1, m2) in cache:
                    return int(cache[(m1, m2)])
                d = int(self.levenshtein_distance(t1, t2))
                cache[(m1, m2)] = d
                cache[(m2, m1)] = d
                return d
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Porovnání transkripcí mezi modely\n")
                f.write("================================\n\n")
                
                for model, text in transcriptions.items():
                    f.write(f"\n{model.upper()}\n")
                    f.write("-" * len(model) + "\n")
                    f.write(f"Délka textu: {len(text)} znaků\n")
                    f.write(f"Text: {text}\n\n")
                
                f.write("\nPorovnání rozdílů mezi modely\n")
                f.write("============================\n\n")
                
                models = list(transcriptions.keys())
                for i, model1 in enumerate(models):
                    for model2 in models[i+1:]:
                        text1 = transcriptions[model1]
                        text2 = transcriptions[model2]
                        distance = _cached_distance(model1, model2, text1, text2)
                        max_len = max(len(text1), len(text2))
                        similarity = self.similarity_from_distance(text1, text2, distance) if max_len else 100.0
                        
                        f.write(f"{model1} vs {model2}\n")
                        f.write(f"Podobnost: {similarity:.2f}%\n")
                        f.write(f"Levenshteinova vzdálenost: {distance}\n\n")
            
            self.logger.info(f"Uloženo porovnání textů do: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Chyba při vytváření vizualizace rozdílů: {str(e)}")
            raise