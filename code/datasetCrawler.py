import json
import os
import pandas as pd
import csv
import time
import re
import requests
from typing import List, Dict, Optional

# --- PROACTIVE SCOPUS CONFIGURATION ---
def _ensure_scopus_config(api_key=None):
    # Forced local config directory inside the project
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.scopus"))
    os.environ['PYBLIOMETRICS_CONFIG_DIR'] = config_dir
    
    config_path = os.path.join(config_dir, "config.ini")
    os.makedirs(config_dir, exist_ok=True)
    
    # Always update or create if key provided
    if not os.path.exists(config_path) or api_key:
        key_to_write = api_key if api_key else ""
        with open(config_path, "w") as f:
            f.write(f"[Authentication]\nAPIKey = {key_to_write}\n")
        print(f"DEBUG: Scopus config ensured at {config_path} with key: {key_to_write[:4]}...")

_ensure_scopus_config()
# --------------------------------------

class DOIEnricher:
    """Enriches dataset records using DOI through Scopus and CrossRef APIs."""
    
    def __init__(self, scopus_api_key=None):
        self.headers = {'User-Agent': 'IRECS-Bibliometric-Tool/1.0 (mailto:jrromero@uco.es)'}
        if scopus_api_key:
            _ensure_scopus_config(scopus_api_key)
            os.environ['PYBLIOMETRICS_API_KEY'] = scopus_api_key

    def fetch_metadata(self, doi):
        """Tries to get metadata from Scopus first, then fallback to CrossRef."""
        if not doi or doi == "N/A" or doi == "undetermined" or doi == "-1":
            return None
        
        # Clean DOI
        doi = doi.strip().replace('https://doi.org/', '').replace('http://doi.org/', '')
        
        result = {
            "Document Title": "N/A",
            "Abstract": "N/A",
            "Year": "N/A",
            "PDF Link": "N/A",
            "doi": doi,
            "nCites": 0,
            "authorCount": 0,
            "aggregationType": "undetermined"
        }

        # 1. Try Scopus
        try:
            # Import inside to ensure env vars are processed
            from pybliometrics.scopus import AbstractRetrieval
            ab = AbstractRetrieval(doi, view="FULL")
            result["Document Title"] = ab.title or "N/A"
            result["Abstract"] = ab.abstract or ab.description or "N/A"
            result["Year"] = ab.coverDate[:4] if ab.coverDate else "N/A"
            result["nCites"] = ab.citedby_count or 0
            result["authorCount"] = len(ab.authors) if ab.authors else 0
            result["aggregationType"] = ab.aggregationType or "undetermined"
            result["PDF Link"] = f"https://www.scopus.com/record/display.uri?eid={ab.eid}"
            return result
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Unauthorized" in msg:
                print(f"Scopus AUTH error for {doi}. Check your API Key.")
            else:
                print(f"Scopus failed for {doi}: {e}. Trying CrossRef...")

        # 2. Fallback to CrossRef (Free)
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()['message']
                result["Document Title"] = data.get('title', ["N/A"])[0]
                
                abstract = data.get('abstract', "N/A")
                if abstract != "N/A":
                    result["Abstract"] = re.sub('<[^<]+?>', '', abstract).strip()
                
                if 'published-print' in data:
                    result["Year"] = data['published-print']['date-parts'][0][0]
                elif 'published-online' in data:
                    result["Year"] = data['published-online']['date-parts'][0][0]
                
                result["nCites"] = data.get('is-referenced-by-count', 0)
                result["authorCount"] = len(data.get('author', []))
                result["PDF Link"] = data.get('URL', "N/A")
                return result
        except Exception as e:
            print(f"CrossRef failed for {doi}: {e}")

        return result

def enrich_dataset(filepath, scopus_api_key=None, progress_callback=None):
    """Detects if a CSV needs enrichment by DOI and creates a new enriched version."""
    print(f"Checking enrichment for {filepath}...")
    
    if scopus_api_key:
        _ensure_scopus_config(scopus_api_key)

    try:
        # Detect delimiter
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            line = f.readline()
            delimiter = ';' if ';' in line and line.count(';') >= line.count(',') else ','
        
        df = pd.read_csv(filepath, sep=delimiter)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return filepath

    # Detect mandatory columns by loose matching
    cols_mapping = {
        "title": next((c for c in df.columns if c.lower() in ["document title", "title"]), None),
        "abstract": next((c for c in df.columns if c.lower() in ["abstract", "description"]), None),
        "doi": next((c for c in df.columns if c.lower() in ["doi"]), None),
        "label": next((c for c in df.columns if c.lower() in ["label", "iscandidate", "candidate", "label_included"]), None)
    }

    if not cols_mapping["doi"]:
        print("No DOI column found. Enrichment impossible.")
        return filepath

    # Check if enrichment is really needed
    needs_enrichment = False
    if not cols_mapping["title"] or not cols_mapping["abstract"]:
        needs_enrichment = True
    else:
        missing_titles = df[cols_mapping["title"]].isna().sum() + (df[cols_mapping["title"]].astype(str).str.lower() == "n/a").sum()
        if missing_titles > len(df) * 0.4:
            needs_enrichment = True

    if not needs_enrichment:
        print("Dataset already contains titles and abstracts.")
        return filepath

    print(f"Starting enrichment process for {len(df)} records...")
    enricher = DOIEnricher(scopus_api_key=scopus_api_key)
    enriched_results = []

    for idx, row in df.iterrows():
        doi = str(row[cols_mapping["doi"]]).strip()
        metadata = enricher.fetch_metadata(doi)
        
        if metadata:
            if cols_mapping["label"]:
                metadata["label"] = row[cols_mapping["label"]]
            else:
                metadata["label"] = "no"
            enriched_results.append(metadata)
        else:
            enriched_results.append({
                "Document Title": "N/A", "Abstract": "N/A", "Year": "N/A",
                "PDF Link": "N/A", "label": "no", "doi": doi,
                "nCites": 0, "authorCount": 0, "aggregationType": "undetermined"
            })
        
        if progress_callback:
            progress_callback(idx + 1, len(df))
        
        time.sleep(0.3)

    # Save new dataset
    enriched_df = pd.DataFrame(enriched_results)
    final_cols = ["Document Title", "Abstract", "Year", "PDF Link", "label", "doi", "nCites", "authorCount", "aggregationType"]
    enriched_df = enriched_df[final_cols]
    
    new_path = filepath.replace(".csv", "_enriched.csv")
    enriched_df.to_csv(new_path, index=False, sep=',')
    
    print(f"Successfully enriched! New file: {new_path}")
    return new_path

# Legacy support
def query(title):
    enricher = DOIEnricher()
    res = enricher.fetch_metadata(title)
    if res:
        return res["authorCount"], res["nCites"], res["aggregationType"], res["doi"], res["Abstract"]
    return 0, 0, 'undetermined', title, 'ERROR'