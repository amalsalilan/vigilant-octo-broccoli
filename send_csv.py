#!/usr/bin/env python3
"""
Simple script to send CSV to LangExtract API and get results
Usage: python send_csv.py your_file.csv
"""

import pandas as pd
import requests
import json
import sys
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
DOC_ID_COLUMN = "doc_id"  # Change this to match your CSV column name

def send_csv(csv_file, output_folder="results"):
    """Send CSV to extraction API"""
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    
    # Read CSV
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} rows")
    
    # Check for doc_id column
    if DOC_ID_COLUMN not in df.columns:
        print(f"\nAvailable columns: {list(df.columns)}")
        print(f"\nError: Column '{DOC_ID_COLUMN}' not found!")
        print("Update DOC_ID_COLUMN in the script to match your column name")
        return
    
    # Check service health
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"\n✓ Connected to service (Model: {health['model_id']})\n")
    except:
        print(f"\n✗ Cannot connect to {API_URL}")
        print("Make sure the FastAPI service is running!")
        return
    
    # Process each row
    all_results = []
    
    for idx, row in df.iterrows():
        doc_id = str(row[DOC_ID_COLUMN])
        print(f"[{idx+1}/{len(df)}] Processing: {doc_id}")
        
        # Prepare row data (exclude doc_id column)
        row_data = {col: str(val) for col, val in row.items() if col != DOC_ID_COLUMN}
        
        # Send to API
        try:
            response = requests.post(
                f"{API_URL}/extract-row",
                json={
                    "doc_id": doc_id,
                    "row": row_data
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                all_results.append(result)
                
                # Save individual result
                output_file = os.path.join(output_folder, f"{doc_id}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Show what was extracted
                fields = result.get('meta', {}).get('fields_processed', [])
                print(f"  ✓ Extracted {len(fields)} fields: {', '.join(fields[:3])}{'...' if len(fields) > 3 else ''}")
                
                # Show artifact URLs if available
                artifacts = result.get('meta', {}).get('artifacts', {})
                if artifacts:
                    print(f"  📄 Reports: {len(artifacts)} HTML files generated")
            else:
                print(f"  ✗ Error {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save consolidated results
    consolidated = os.path.join(output_folder, "all_results.json")
    with open(consolidated, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Processed {len(all_results)}/{len(df)} rows")
    print(f"Results saved to: {output_folder}/")
    print(f"  - Individual files: {output_folder}/<doc_id>.json")
    print(f"  - Consolidated: {consolidated}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_csv.py <csv_file>")
        print("Example: python send_csv.py multi_doc_contracts.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    send_csv(csv_file)