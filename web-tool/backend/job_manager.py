import uuid
import threading
import sys
import os
import io
import time
import logging
from typing import Dict, Any, List

# Add the parent directory to sys.path to import from 'code'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code')))

from main import launchExperiment
from g3pEngine import ReplacementStrategy, ClassificationStrategy

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.enrichments: Dict[str, Dict[str, Any]] = {}
        self.pause_events: Dict[str, threading.Event] = {}

    def create_enrichment_task(self, filename: str) -> str:
        task_id = str(uuid.uuid4())
        self.enrichments[task_id] = {
            "status": "running",
            "progress": 0,
            "current": 0,
            "total": 0,
            "filename": filename,
            "result_path": None,
            "error": None
        }
        return task_id

    def run_enrichment(self, task_id: str, filepath: str, api_key: str = None):
        from datasetCrawler import enrich_dataset
        task = self.enrichments[task_id]
        
        def progress_callback(current, total):
            task["current"] = current
            task["total"] = total
            task["progress"] = round((current / total) * 100)

        try:
            new_path = enrich_dataset(filepath, scopus_api_key=api_key, progress_callback=progress_callback)
            task["status"] = "completed"
            task["progress"] = 100
            task["result_path"] = new_path
            
            # Auto-cleanup: if we created a new enriched file, delete the useless original
            if new_path != filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logging.info(f"Cleanup: Deleted original non-enriched file {filepath}")
                except Exception as cleanup_err:
                    logging.warning(f"Failed to cleanup original file: {cleanup_err}")
                    
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)

    def start_enrichment(self, task_id: str, filepath: str, api_key: str = None):
        thread = threading.Thread(target=self.run_enrichment, args=(task_id, filepath, api_key))
        thread.start()

    def create_job(self, config: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        self.pause_events[job_id] = threading.Event()
        self.jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "logs": [],
            "results": None,
            "best_rules": None,
            "selected_papers": [],
            "fitness_history": [],
            "config": config,
            "start_time": None,
            "end_time": None,
            "error": None
        }
        return job_id

    def run_experiment(self, job_id: str):
        job = self.jobs[job_id]
        job["status"] = "running"
        job["start_time"] = time.time()
        
        # Capture stdout for logs
        log_stream = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = log_stream

        try:
            config = job["config"]

            # Resolve grammar path independently of the process cwd.
            code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code'))

            def resolve_grammar_path(raw_path: str) -> str:
                if not raw_path:
                    raw_path = "assets/ml.xml"

                # Absolute path can be used directly.
                if os.path.isabs(raw_path):
                    return raw_path

                # Common relative forms accepted from API/UI.
                candidates = [
                    raw_path,
                    os.path.join(code_dir, raw_path),
                ]

                # Handle inputs like "code/assets/ml.xml".
                code_prefix = f"code{os.sep}"
                if raw_path.startswith("code/") or raw_path.startswith(code_prefix):
                    trimmed = raw_path.split("code/", 1)[-1] if "code/" in raw_path else raw_path[len(code_prefix):]
                    candidates.append(os.path.join(code_dir, trimmed))

                for candidate in candidates:
                    candidate_abs = os.path.abspath(candidate)
                    if os.path.exists(candidate_abs):
                        return candidate_abs

                # Fallback to code directory to keep previous default behavior predictable.
                return os.path.abspath(os.path.join(code_dir, raw_path))
            
            # Generate unique filenames for this job
            log_file = f"log_{job_id}.txt"
            rules_file = f"best_rules_{job_id}.txt"

            # Extract parameters
            params = {
                "datasetFilePath": config.get("datasetFilePath"),
                "grammarFilePath": resolve_grammar_path(config.get("grammarFilePath", "assets/ml.xml")),
                "nFolds": int(config.get("nFolds", 2)),
                "maxGenerations": int(config.get("maxGenerations", 10)),
                "populationSize": int(config.get("populationSize", 30)),
                "crossProb": float(config.get("crossProb", 0.9)),
                "mutationProb": float(config.get("mutationProb", 0.1)),
                "fitnessThreshold": float(config.get("fitnessThreshold", 0.4)),
                "replacementStrategy": ReplacementStrategy[config.get("replacementStrategy", "NEWPOPULATION")],
                "classificationStrategy": ClassificationStrategy[config.get("classificationStrategy", "CBA")],
                "positiveWeight": float(config.get("positiveWeight", 1.5)),
                "seed": int(config.get("seed", 1)),
                "logFilePath": log_file,
                "bestRulesFilePath": rules_file,
                "vocabStrategy": config.get("vocabStrategy", "ALL"),
                "extraTerms": config.get("extraTerms", ""),
                "pause_event": self.pause_events[job_id]
            }

            def update_progress(p):
                job["progress"] = min(99, round(p))
                # Update rules from file in real-time
                if os.path.exists(rules_file):
                    try:
                        with open(rules_file, "r") as f:
                            job["best_rules"] = f.read()
                    except Exception:
                        pass

            # Run experiment
            results, histories, selected_papers = launchExperiment(**params, progress_callback=update_progress)
            job["results"] = results
            job["selected_papers"] = selected_papers
            job["progress"] = 100
            job["fitness_history"] = histories
            job["status"] = "completed"
            
            # Read unique best rules file
            if os.path.exists(rules_file):
                with open(rules_file, "r") as f:
                    job["best_rules"] = f.read()
            
            # Cleanup temporary files
            try:
                if os.path.exists(log_file): os.remove(log_file)
                if os.path.exists(rules_file): os.remove(rules_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup temporary files: {e}")
            
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            print(f"Error in experiment: {e}")
        finally:
            sys.stdout = original_stdout
            job["end_time"] = time.time()
            job["logs"] = log_stream.getvalue().splitlines()

    def start_job(self, job_id: str):
        thread = threading.Thread(target=self.run_experiment, args=(job_id,))
        thread.start()

    def toggle_pause(self, job_id: str):
        if job_id not in self.jobs: return
        event = self.pause_events[job_id]
        if event.is_set():
            event.clear()
            self.jobs[job_id]["status"] = "running"
        else:
            event.set()
            self.jobs[job_id]["status"] = "paused"

    def update_job_config(self, job_id: str, new_config: Dict[str, Any]):
        if job_id not in self.jobs: return
        # Merge new config into existing config
        self.jobs[job_id]["config"].update(new_config)

    def _sanitize_data(self, data: Any) -> Any:
        """Recursively replaces NaN/Inf with None to ensure JSON compliance."""
        import math
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(v) for v in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return None
        return data

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            return {"status": "not_found"}
        return self._sanitize_data(self.jobs[job_id])

manager = JobManager()
