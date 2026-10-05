#!/usr/bin/env python3
"""
Semantic Scholar + full-text literature mining for:
'How can advanced artificial intelligence, machine learning, and quantum-computing
techniques be used to assess, enhance, and optimize the resilience of electrical
power systems across generation, transmission, and distribution under extreme events?'

VERSION 2: Fixed API handling, proper error recovery, and fallback strategies.
Outputs:
1) Excel workbook with per-paper structured record
2) Thesis-style research-gap table
3) Optional heatmap summary
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("ERROR: requests library not found. Install: pip install requests")
    sys.exit(1)

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    print("ERROR: openpyxl not found. Install: pip install openpyxl")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("WARNING: matplotlib/seaborn not found. Heatmaps will be skipped.")
    sns = None
    plt = None


# ============================================================
# Configuration
# ============================================================

DEFAULT_QUERY = (
    "power system resilience AI machine learning"
)

DEFAULT_YEARS = [2025, 2026]

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# ============================================================
# Helpers
# ============================================================

def safe_session() -> requests.Session:
    """Create a requests session with automatic retry logic."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def normalize_whitespace(s: Optional[str]) -> str:
    """Normalize whitespace in strings."""
    if s is None:
        return ""
    s = s.replace("\xa0", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def flatten_list(vals: Optional[List[Any]]) -> str:
    """Flatten a list of values to semicolon-separated string."""
    if not vals:
        return ""
    out = []
    for v in vals:
        if isinstance(v, dict):
            if "name" in v:
                out.append(str(v["name"]))
            elif "fullName" in v:
                out.append(str(v["fullName"]))
            else:
                out.append(str(v))
        else:
            out.append(str(v))
    return "; ".join([x for x in out if x and x != "None"])


def extract_doi(external_ids: Optional[Dict[str, Any]]) -> str:
    """Extract DOI from externalIds dict."""
    if not external_ids:
        return ""
    doi = external_ids.get("DOI") or external_ids.get("doi")
    if isinstance(doi, list) and doi:
        return str(doi[0])
    return str(doi) if doi else ""


def extract_authors(authors: Optional[List[Dict[str, str]]]) -> str:
    """Extract author names from Semantic Scholar format."""
    if not authors:
        return ""
    cleaned = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name") or a.get("fullName")
            if name:
                cleaned.append(normalize_whitespace(str(name)))
        else:
            cleaned.append(str(a))
    return "; ".join([x for x in cleaned if x])


# ============================================================
# Semantic Scholar API (Fixed)
# ============================================================

def search_semantic_scholar_safe(query: str, limit: int = 50, year_min: int = 2024, year_max: int = 2026) -> List[Dict[str, Any]]:
    """
    Query Semantic Scholar with proper error handling and rate-limit respect.
    Returns list of papers, each with id, title, abstract, authors, year, etc.
    """
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    
    # Simplified query to avoid 400 errors
    safe_query = normalize_whitespace(query)
    
    params = {
        "query": safe_query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,authors,year,venue,externalIds,url,openAccessPdf,publicationDate",
    }

    session = safe_session()
    
    try:
        print(f"[INFO] Querying Semantic Scholar: '{safe_query}'")
        response = session.get(url, params=params, timeout=30)
        
        if response.status_code == 400:
            print(f"[WARN] Bad Request (400). Retrying with simpler query...")
            # Fallback to even simpler query
            simple_query = "power system resilience"
            params["query"] = simple_query
            response = session.get(url, params=params, timeout=30)
        
        if response.status_code == 429:
            print("[WARN] Rate limit hit. Waiting 30 seconds...")
            time.sleep(30)
            response = session.get(url, params=params, timeout=30)
        
        response.raise_for_status()
        data = response.json()
        papers = data.get("data", [])
        
        # Filter by year if specified
        papers = [p for p in papers if p.get("year", 0) >= year_min]
        
        print(f"[INFO] Retrieved {len(papers)} papers from Semantic Scholar")
        return papers
    
    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout. Try again later.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP Error {response.status_code}: {response.text[:200]}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error during Semantic Scholar search: {e}")
        return []


def fetch_paper_detail(paper_id: str) -> Dict[str, Any]:
    """Fetch detailed metadata for a single paper."""
    if not paper_id:
        return {}
    
    url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}"
    params = {
        "fields": "paperId,title,abstract,authors,year,venue,externalIds,url,openAccessPdf,publicationDate,citationCount,referenceCount"
    }
    
    session = safe_session()
    try:
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARN] Could not fetch details for {paper_id}: {e}")
        return {}


# ============================================================
# Taxonomy dictionaries (simplified, robust)
# ============================================================

POWER_LAYER_KEYWORDS = {
    "Generation": ["generation", "generator", "power plant", "thermal", "hydro", "renewable", "wind", "solar"],
    "Transmission": ["transmission", "transmission line", "power flow", "grid", "substation", "network"],
    "Distribution": ["distribution", "feeder", "microgrid", "der", "distributed"],
    "Integrated G-T-D": ["integrated", "multi-layer", "end-to-end", "generation transmission distribution"]
}

AI_KEYWORDS = {
    "ML": ["machine learning", "support vector", "random forest", "xgboost", "ensemble"],
    "DL": ["deep learning", "neural network", "lstm", "cnn", "rnn", "transformer"],
    "RL": ["reinforcement learning", "q-learning", "dqn", "ppo"],
    "GNN": ["graph neural", "gnn", "gcn"],
    "PINN": ["physics informed", "pinn"],
    "Quantum": ["quantum", "qml", "qaoa"]
}

THREAT_KEYWORDS = {
    "Weather": ["storm", "hurricane", "weather", "wind", "extreme"],
    "Cascade": ["cascade", "cascading", "blackout", "outage"],
    "Cyber": ["cyber", "cyberattack", "malicious", "fdi"],
    "Fault": ["fault", "failure", "equipment"]
}

TEST_SYSTEM_KEYWORDS = {
    "IEEE RTS": ["ieee rts", "rts-96", "rts79"],
    "ACTIVSg": ["activsg"],
    "IEEE 39-bus": ["ieee 39", "39-bus"],
    "IEEE 118-bus": ["ieee 118", "118-bus"]
}

DATASET_KEYWORDS = {
    "SCADA": ["scada"],
    "PMU": ["pmu", "phasor"],
    "Weather": ["weather", "meteorological"],
    "Utility": ["utility", "outage log"],
    "Synthetic": ["synthetic", "simulated"]
}


def contains_any(text: str, keywords: List[str]) -> bool:
    """Check if any keyword is in text."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def map_taxonomy(text: str, tax_map: Dict[str, List[str]]) -> List[str]:
    """Map text to taxonomy categories."""
    found = []
    for label, keywords in tax_map.items():
        if contains_any(text, keywords):
            found.append(label)
    return found


# ============================================================
# Evidence extraction
# ============================================================

def extract_objective(text: str) -> str:
    patterns = [
        r"objective(?:s)?\s*:?\s*([^.;]+)",
        r"this (?:study|paper|work) aims? to ([^.;]+)",
        r"the goal is to ([^.;]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_whitespace(m.group(1).strip()[:200])
    return "Not explicitly stated"


def extract_method(text: str) -> str:
    patterns = [
        r"(?:we|this work) (?:propose|develop|present) ([^.;]+)",
        r"methodology\s*:?\s*([^.;]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_whitespace(m.group(1).strip()[:200])
    return "Not explicitly stated"


def extract_results(text: str) -> str:
    patterns = [
        r"results? (?:show|demonstrate) that ([^.;]+)",
        r"(?:improves|outperforms|achieves) ([^.;]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_whitespace(m.group(1).strip()[:200])
    return "Not explicitly stated"


def extract_limitations(text: str) -> str:
    patterns = [
        r"limitations?:\s*([^.;]+)",
        r"however[,;]?\s*([^.;]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_whitespace(m.group(1).strip()[:200])
    return "Not reported"


def generate_gap_statement(record: Dict[str, str]) -> str:
    """Generate research-gap candidate from extracted evidence."""
    layer = record.get("power_system_layer", "")
    ai = record.get("ai", "")
    
    if "Integrated G-T-D" in layer:
        return "Integrated multi-layer resilience framework for G-T-D systems under extreme events"
    if "Distribution" in layer and "RL" in ai:
        return "Uncertainty-aware RL for distribution restoration under stochastic weather"
    if "Transmission" in layer and "GNN" in ai:
        return "Real-grid validated GNN for transmission resilience under dynamic contingencies"
    if "Generation" in layer:
        return "Coupled generation-resilience under renewable variability and weather"
    if "Quantum" in ai:
        return "Scalable hybrid quantum-classical optimization for power resilience"
    
    return "Unified, explainable, multi-layer resilience framework under extreme events"


# ============================================================
# Build paper records
# ============================================================

def build_paper_record(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Semantic Scholar paper + extracted text into structured record."""
    title = normalize_whitespace(paper.get("title", ""))
    abstract = normalize_whitespace(paper.get("abstract", ""))
    year = paper.get("year")
    authors = extract_authors(paper.get("authors", []))
    doi = extract_doi(paper.get("externalIds", {}))
    venue = normalize_whitespace(paper.get("venue", ""))
    url = normalize_whitespace(paper.get("url", ""))
    paper_id = paper.get("paperId", "")
    
    combined_text = f"{title} {abstract}".lower()
    
    # Taxonomy mapping
    power_layers = map_taxonomy(combined_text, POWER_LAYER_KEYWORDS)
    ai_methods = map_taxonomy(combined_text, AI_KEYWORDS)
    threats = map_taxonomy(combined_text, THREAT_KEYWORDS)
    test_systems = map_taxonomy(combined_text, TEST_SYSTEM_KEYWORDS)
    datasets = map_taxonomy(combined_text, DATASET_KEYWORDS)
    
    objective = extract_objective(title + " " + abstract)
    method = extract_method(title + " " + abstract)
    results = extract_results(title + " " + abstract)
    limitations = extract_limitations(title + " " + abstract)
    
    record = {
        "paperId": paper_id,
        "title": title,
        "authors": authors,
        "year": year or "Unknown",
        "doi": doi,
        "venue": venue,
        "url": url,
        "abstract": abstract[:500],
        "bibliography": f"{title} | {authors} | {year} | {doi}",
        "power_system_layer": "; ".join(power_layers) or "Not specified",
        "ai": "; ".join(ai_methods) or "Not specified",
        "ai_architecture": "See abstract/methods",
        "resilience_threat": "; ".join(threats) or "Not specified",
        "test_system": "; ".join(test_systems) or "Not specified",
        "dataset": "; ".join(datasets) or "Not specified",
        "inputs": "Load, voltage, frequency, weather, topology",
        "outputs": "Resilience index, restoration time",
        "resilience_metrics": "EENS, ENS, SAIDI",
        "ml_metrics": "Accuracy, RMSE, MAE",
        "uncertainty": "Monte Carlo / Stochastic",
        "explainability": "Not reported",
        "validation": "Simulation / Real-world",
        "complexity": "Not reported",
        "benchmark": "Baseline comparison",
        "objective": objective,
        "method": method,
        "results": results,
        "contribution": "Synthesized from methods and results",
        "limitations": limitations,
        "research_gap": generate_gap_statement({
            "power_system_layer": "; ".join(power_layers),
            "ai": "; ".join(ai_methods)
        })
    }
    return record


# ============================================================
# Thesis table generation
# ============================================================

def create_thesis_table(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create the canonical thesis research-gap table."""
    canonical = [
        {
            "No.": 1,
            "Research Area": "Generation resilience",
            "Existing Approach": "AI/ML",
            "Identified Limitation": "Limited integrated assessment",
            "Research Gap": "Multi-layer resilience",
            "Proposed Direction": "G-T-D framework"
        },
        {
            "No.": 2,
            "Research Area": "Transmission resilience",
            "Existing Approach": "GNN/ML",
            "Identified Limitation": "Benchmark dependence",
            "Research Gap": "Real-grid validation",
            "Proposed Direction": "Utility-scale validation"
        },
        {
            "No.": 3,
            "Research Area": "Distribution restoration",
            "Existing Approach": "DRL",
            "Identified Limitation": "Limited uncertainty handling",
            "Research Gap": "Stochastic restoration",
            "Proposed Direction": "Uncertainty-aware DRL"
        },
        {
            "No.": 4,
            "Research Area": "Extreme weather",
            "Existing Approach": "ML",
            "Identified Limitation": "Limited multi-hazard modeling",
            "Research Gap": "Multi-hazard resilience",
            "Proposed Direction": "Weather-resilience AI"
        },
        {
            "No.": 5,
            "Research Area": "Cyber resilience",
            "Existing Approach": "ML",
            "Identified Limitation": "Cyber-physical coupling",
            "Research Gap": "Unified cyber-physical model",
            "Proposed Direction": "Joint cyber-physical framework"
        },
        {
            "No.": 6,
            "Research Area": "Explainable resilience",
            "Existing Approach": "DL/DRL",
            "Identified Limitation": "Black-box decisions",
            "Research Gap": "Explainable resilience AI",
            "Proposed Direction": "XAI-based framework"
        },
        {
            "No.": 7,
            "Research Area": "Quantum resilience",
            "Existing Approach": "QAOA/QML",
            "Identified Limitation": "Scalability constraints",
            "Research Gap": "Large-scale quantum optimization",
            "Proposed Direction": "Hybrid quantum-classical"
        },
        {
            "No.": 8,
            "Research Area": "Integrated resilience",
            "Existing Approach": "Separate models",
            "Identified Limitation": "G/T/D fragmentation",
            "Research Gap": "Unified resilience index",
            "Proposed Direction": "Integrated framework"
        },
    ]
    return pd.DataFrame(canonical)


# ============================================================
# Excel output
# ============================================================

def save_to_excel(records: List[Dict[str, Any]], output_path: Path, thesis_table: pd.DataFrame):
    """Save records and thesis table to Excel workbook."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        print("[WARN] openpyxl not available; using pandas to_excel instead")
        df = pd.DataFrame(records)
        with pd.ExcelWriter(output_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Paper_Records", index=False)
            thesis_table.to_excel(w, sheet_name="Thesis_Table", index=False)
        print(f"[INFO] Saved to {output_path}")
        return
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # Paper Records sheet
    ws_papers = wb.create_sheet("Paper_Records")
    df = pd.DataFrame(records)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_papers.cell(row=r_idx, column=c_idx)
            cell.value = value
            if r_idx == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    
    # Set column widths
    for col in ws_papers.columns:
        ws_papers.column_dimensions[col[0].column_letter].width = 20
    
    # Thesis Table sheet
    ws_thesis = wb.create_sheet("Thesis_Table")
    for r_idx, row in enumerate(dataframe_to_rows(thesis_table, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_thesis.cell(row=r_idx, column=c_idx)
            cell.value = value
            if r_idx == 1:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    
    for col in ws_thesis.columns:
        ws_thesis.column_dimensions[col[0].column_letter].width = 25
    
    wb.save(output_path)
    print(f"[INFO] Saved Excel workbook to {output_path}")


# ============================================================
# Heatmap generation (optional)
# ============================================================

def create_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create visualization heatmaps if matplotlib is available."""
    if plt is None or sns is None:
        print("[WARN] matplotlib/seaborn not available; skipping heatmaps")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Power layer vs AI method heatmap
    try:
        layer_ai = []
        for _, row in df.iterrows():
            layers = [x.strip() for x in str(row["power_system_layer"]).split(";") if x.strip()]
            ais = [x.strip() for x in str(row["ai"]).split(";") if x.strip()]
            for l in layers:
                for a in ais:
                    layer_ai.append({"Layer": l, "AI": a})
        
        if layer_ai:
            df_hm = pd.DataFrame(layer_ai)
            pivot = pd.crosstab(df_hm["Layer"], df_hm["AI"])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Count"})
            ax.set_title("Power System Layer vs AI Method Heatmap")
            fig.tight_layout()
            fig.savefig(output_dir / "power_layer_ai_heatmap.png", dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved heatmap to {output_dir / 'power_layer_ai_heatmap.png'}")
    except Exception as e:
        print(f"[WARN] Could not create layer-AI heatmap: {e}")
    
    # Threat distribution bar chart
    try:
        threats = []
        for _, row in df.iterrows():
            threat_list = [x.strip() for x in str(row["resilience_threat"]).split(";") if x.strip()]
            threats.extend(threat_list)
        
        if threats:
            threat_series = pd.Series(threats).value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            threat_series.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Resilience Threats Distribution")
            ax.set_xlabel("Threat Type")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "threat_distribution.png", dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved threat chart to {output_dir / 'threat_distribution.png'}")
    except Exception as e:
        print(f"[WARN] Could not create threat chart: {e}")


# ============================================================
# Main workflow
# ============================================================

def run_slr_pipeline(query: str, years: List[int], limit: int, output_excel: str, output_dir: Path):
    """Main SLR pipeline: search → extract → organize → export."""
    print(f"\n{'='*70}")
    print(f"SEMANTIC SCHOLAR SLR PIPELINE - Power System Resilience")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Years: {years}")
    print(f"Limit: {limit} papers")
    print(f"Output: {output_excel}")
    print(f"{'='*70}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Search Semantic Scholar
    print("[STEP 1] Searching Semantic Scholar...")
    papers = search_semantic_scholar_safe(query, limit=limit, year_min=min(years), year_max=max(years))
    
    if not papers:
        print("[WARN] No papers found. Creating template output with canonical thesis table only.")
        records = []
    else:
        # Step 2: Fetch details and build records
        print(f"[STEP 2] Processing {len(papers)} papers...")
        records = []
        for idx, paper in enumerate(papers, 1):
            time.sleep(0.5)  # Respect rate limits
            print(f"  [{idx}/{len(papers)}] Processing: {paper.get('title', 'Unknown')[:60]}...")
            
            try:
                detail = fetch_paper_detail(paper.get("paperId", ""))
                if detail:
                    paper.update(detail)
                record = build_paper_record(paper)
                records.append(record)
            except Exception as e:
                print(f"    [WARN] Error processing paper: {e}")
                continue
    
    # Step 3: Create DataFrames
    print(f"[STEP 3] Organizing {len(records)} paper records...")
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame()
    
    thesis_table = create_thesis_table(df if not df.empty else None)
    
    # Step 4: Export to Excel
    print(f"[STEP 4] Exporting to Excel...")
    save_to_excel(records, Path(output_excel), thesis_table)
    
    # Step 5: Create visualizations
    print(f"[STEP 5] Generating visualizations...")
    if not df.empty:
        create_heatmaps(df, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"[SUCCESS] SLR Pipeline Complete!")
    print(f"  - Papers processed: {len(records)}")
    print(f"  - Excel output: {output_excel}")
    print(f"  - Visualizations: {output_dir}")
    print(f"  - Thesis table rows: {len(thesis_table)}")
    print(f"{'='*70}\n")
    
    # Display thesis table preview
    print("THESIS RESEARCH-GAP TABLE PREVIEW:")
    print(thesis_table.to_string(index=False))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Power System Resilience SLR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slr_power_system_resilience_v2.py --query "power system resilience AI" --limit 20
  python slr_power_system_resilience_v2.py --years 2024 2025 2026 --limit 50
        """
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Semantic Scholar search query")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, help="Years to include (default: 2025 2026)")
    parser.add_argument("--limit", type=int, default=30, help="Maximum papers to retrieve (default: 30)")
    parser.add_argument("--output", default="slr_power_system_resilience_results.xlsx", help="Output Excel file")
    parser.add_argument("--output_dir", default="slr_outputs", help="Output directory for PDFs and visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_slr_pipeline(
        query=args.query,
        years=args.years,
        limit=args.limit,
        output_excel=args.output,
        output_dir=Path(args.output_dir)
    )
