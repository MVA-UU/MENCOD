#!/usr/bin/env python3
"""
Verify and Complete Citation Extraction

This script verifies an existing file of extracted cited documents against
what should be available from a synergy dataset, identifies missing documents,
and fetches them with robust rate limiting to ensure complete data coverage.

Usage:
    python verify_and_complete_extraction.py

The script will:
1. Load your existing extracted records file
2. Load the synergy dataset to determine what documents should exist
3. Identify missing documents
4. Fetch missing documents with robust rate limiting
5. Save missing records individually to temp_records/
6. Create a complete merged file with all data
"""

import asyncio
import aiohttp
import pandas as pd
import json
import os
import sys
from typing import Set, List, Dict, Any, Optional
from pathlib import Path

# Import synergy-dataset package
from synergy_dataset import Dataset, iter_datasets


class OpenAlexClient:
    """
    Async client for OpenAlex API with robust rate limiting and error handling.
    
    Features:
    - Respects OpenAlex 10 concurrent request limit via semaphore
    - Exponential backoff for rate limiting (429 errors)
    - Multiple retry attempts (up to 5) for temporary failures
    - Permanent failure detection (404, repeated failures)
    - Tracks failed requests to ensure no data loss
    - Distinguishes between temporary (retry) and permanent (skip) failures
    """
    
    def __init__(self, max_concurrent_requests: int = 10):
        self.base_url = "https://api.openalex.org"
        self.max_concurrent = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        
        self.headers = {
            'User-Agent': 'mailto:your-email@domain.com',
            'Accept': 'application/json'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_work_details(self, openalex_id: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
        """Get detailed information for a single work with robust retry logic."""
        async with self.semaphore:
            clean_id = openalex_id.replace('https://openalex.org/', '')
            if not clean_id.startswith('W'):
                clean_id = f"W{clean_id}" if clean_id.isdigit() else clean_id
            
            url = f"{self.base_url}/works/{clean_id}"
            
            for attempt in range(max_retries + 1):
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data
                        elif response.status == 429:
                            wait_time = min(2 ** attempt, 60)
                            print(f"Rate limited for {clean_id} (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status == 404:
                            print(f"Document {clean_id} not found (404)")
                            return None
                        elif response.status >= 500:
                            wait_time = min(2 ** attempt, 30)
                            print(f"Server error {response.status} for {clean_id} (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            print(f"Failed to fetch {clean_id}: HTTP {response.status} (attempt {attempt + 1}/{max_retries + 1})")
                            if attempt < max_retries:
                                await asyncio.sleep(1)
                                continue
                            return None
                        
                except Exception as e:
                    print(f"Error fetching {openalex_id} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    return None
            
            await asyncio.sleep(0.1)
            
        print(f"PERMANENT FAILURE: Could not fetch {clean_id} after {max_retries + 1} attempts")
        return None
    
    async def get_works_batch_with_retry_queue(self, openalex_ids: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
        """Get details for multiple works with retry queue for failed requests."""
        tasks = [self.get_work_details(openalex_id) for openalex_id in openalex_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        failed_ids = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception for {openalex_ids[i]}: {result}")
                failed_ids.append(openalex_ids[i])
            elif result is not None:
                successful_results.append(result)
            else:
                failed_ids.append(openalex_ids[i])
        
        return successful_results, failed_ids


def get_existing_file():
    """Prompt user to select existing extracted records file."""
    print("\n=== Select Existing Extracted Records File ===")
    
    # Look for CSV files in data and external_data directories
    data_dir = Path('data')
    external_data_dir = data_dir / 'external_data'
    
    csv_files = []
    
    # Check data directory
    if data_dir.exists():
        csv_files.extend(list(data_dir.glob('*.csv')))
    
    # Check external_data directory
    if external_data_dir.exists():
        csv_files.extend(list(external_data_dir.glob('*.csv')))
    
    if not csv_files:
        print("No CSV files found in data/ or data/external_data/")
        file_path = input("Please enter the full path to your extracted records file: ").strip()
        return Path(file_path)
    
    print("Found CSV files:")
    for i, file_path in enumerate(csv_files):
        print(f"{i+1}. {file_path}")
    
    print(f"{len(csv_files)+1}. Enter custom path")
    
    while True:
        try:
            selection = int(input(f"\nSelect file (1-{len(csv_files)+1}): "))
            if 1 <= selection <= len(csv_files):
                return csv_files[selection-1]
            elif selection == len(csv_files)+1:
                file_path = input("Enter full path to your file: ").strip()
                return Path(file_path)
            else:
                print(f"Please enter a number between 1 and {len(csv_files)+1}")
        except ValueError:
            print("Please enter a valid number")


def prompt_synergy_dataset_selection() -> str:
    """Prompt user to select a synergy dataset."""
    try:
        datasets = []
        print("Loading available datasets...")
        for d in iter_datasets():
            datasets.append(d.name)
        
        if not datasets:
            raise ValueError("No datasets available")
            
    except Exception as e:
        print(f"Error getting available datasets: {e}")
        datasets = [
            "Appenzeller-Herzog_2019",
            "Hall_2012", 
            "Jeyaraman_2020",
            "van_der_Valk_2021"
        ]
    
    print(f"\nAvailable synergy datasets ({len(datasets)} total):")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset}")
    
    while True:
        try:
            selection = int(input("\nSelect dataset (enter number): "))
            if 1 <= selection <= len(datasets):
                return datasets[selection-1]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")


def load_existing_records(file_path: Path) -> Set[str]:
    """Load existing extracted records and return set of OpenAlex IDs."""
    print(f"Loading existing records from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        
        # Look for OpenAlex ID column (various possible names)
        id_columns = ['openalex_id', 'id', 'work_id', 'openalex_work_id']
        id_column = None
        
        for col in id_columns:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            print("Available columns:", df.columns.tolist())
            id_column = input("Enter the column name that contains OpenAlex IDs: ").strip()
        
        # Extract clean IDs
        existing_ids = set()
        for idx, row in df.iterrows():
            openalex_id = str(row[id_column])
            if openalex_id and openalex_id != 'nan':
                clean_id = openalex_id.replace('https://openalex.org/', '')
                existing_ids.add(clean_id)
        
        print(f"Loaded {len(existing_ids)} existing records")
        print(f"Sample IDs: {list(existing_ids)[:3]}")
        return existing_ids
        
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_expected_ids_from_synergy(dataset_name: str) -> tuple[Set[str], Set[str]]:
    """Get expected referenced works from synergy dataset."""
    print(f"Loading synergy dataset: {dataset_name}")
    
    try:
        dataset = Dataset(dataset_name)
        data_dict = dataset.to_dict(["title", "referenced_works"])
        print(f"Loaded {len(data_dict)} documents from synergy dataset")
        
        existing_ids = set()
        all_referenced_works = set()
        
        for openalex_url, doc_data in data_dict.items():
            clean_id = openalex_url.replace('https://openalex.org/', '')
            existing_ids.add(clean_id)
            
            referenced_works = doc_data.get('referenced_works', [])
            if referenced_works and isinstance(referenced_works, list):
                for ref_work in referenced_works:
                    if isinstance(ref_work, str):
                        clean_ref_id = ref_work.replace('https://openalex.org/', '')
                        all_referenced_works.add(clean_ref_id)
        
        print(f"Dataset contains {len(existing_ids)} papers")
        print(f"Total referenced works: {len(all_referenced_works)}")
        
        return existing_ids, all_referenced_works
        
    except Exception as e:
        print(f"Error loading synergy dataset: {e}")
        raise


def prepare_work_data_for_csv(work_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant fields from OpenAlex work data for CSV storage."""
    row = {
        'openalex_id': work_data.get('id', '').replace('https://openalex.org/', ''),
        'title': work_data.get('title', ''),
        'abstract': work_data.get('abstract', ''),
        'publication_year': work_data.get('publication_year'),
        'publication_date': work_data.get('publication_date'),
        'type': work_data.get('type'),
        'doi': work_data.get('doi', ''),
        'is_oa': work_data.get('open_access', {}).get('is_oa', False),
        'cited_by_count': work_data.get('cited_by_count', 0)
    }
    
    # Extract author information
    authors = []
    if 'authorships' in work_data and work_data['authorships']:
        for authorship in work_data['authorships'][:5]:
            author = authorship.get('author', {})
            if author.get('display_name'):
                authors.append(author['display_name'])
    row['authors'] = '; '.join(authors) if authors else ''
    
    # Extract venue information
    primary_location = work_data.get('primary_location', {})
    if primary_location:
        source = primary_location.get('source', {})
        row['journal'] = source.get('display_name', '') if source else ''
        row['publisher'] = source.get('host_organization_name', '') if source else ''
    else:
        row['journal'] = ''
        row['publisher'] = ''
    
    # Extract concepts
    concepts = []
    if 'concepts' in work_data and work_data['concepts']:
        for concept in work_data['concepts'][:10]:
            if concept.get('display_name') and concept.get('score', 0) > 0.3:
                concepts.append(f"{concept['display_name']} ({concept['score']:.2f})")
    row['concepts'] = '; '.join(concepts) if concepts else ''
    
    # Extract referenced works
    referenced_works = []
    if 'referenced_works' in work_data and work_data['referenced_works']:
        for ref_url in work_data['referenced_works']:
            if isinstance(ref_url, str):
                clean_id = ref_url.replace('https://openalex.org/', '')
                referenced_works.append(clean_id)
    
    row['referenced_works'] = '; '.join(referenced_works) if referenced_works else ''
    row['referenced_works_count'] = len(referenced_works)
    
    # Extract mesh terms
    mesh_terms = []
    if 'mesh' in work_data and work_data['mesh']:
        for mesh in work_data['mesh'][:10]:
            if mesh.get('descriptor_name'):
                mesh_terms.append(mesh['descriptor_name'])
    row['mesh_terms'] = '; '.join(mesh_terms) if mesh_terms else ''
    
    return row


async def fetch_missing_documents(missing_ids: Set[str], temp_dir: Path) -> tuple[int, Set[str]]:
    """Fetch missing documents and save to temp directory."""
    if not missing_ids:
        return 0, set()
    
    print(f"\n=== Fetching {len(missing_ids)} Missing Documents ===")
    
    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary records will be saved to: {temp_dir}")
    
    missing_list = list(missing_ids)
    batch_size = 50
    total_fetched = 0
    all_failed_ids = set()
    
    async with OpenAlexClient() as client:
        for i in range(0, len(missing_list), batch_size):
            batch = missing_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(missing_list) + batch_size - 1)//batch_size
            
            print(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} works)")
            
            works_data, failed_ids = await client.get_works_batch_with_retry_queue(batch)
            
            # Save each work immediately
            batch_saved = 0
            for work_data in works_data:
                if work_data:
                    try:
                        work_id = work_data.get('id', '').replace('https://openalex.org/', '')
                        if work_id:
                            record_file = temp_dir / f"{work_id}.json"
                            with open(record_file, 'w', encoding='utf-8') as f:
                                json.dump(work_data, f, indent=2, ensure_ascii=False)
                            batch_saved += 1
                            total_fetched += 1
                    except Exception as e:
                        print(f"Error saving work data: {e}")
                        continue
            
            if failed_ids:
                all_failed_ids.update(failed_ids)
                print(f"Failed to fetch {len(failed_ids)} works in this batch")
            
            print(f"Saved {batch_saved} works from this batch (Total: {total_fetched}/{len(missing_list)})")
            await asyncio.sleep(0.5)
    
    print(f"Finished fetching! Total works saved: {total_fetched}")
    
    if all_failed_ids:
        print(f"⚠️  WARNING: {len(all_failed_ids)} documents could not be fetched")
        failed_file = temp_dir / 'failed_ids.txt'
        with open(failed_file, 'w') as f:
            for failed_id in sorted(all_failed_ids):
                f.write(f"{failed_id}\n")
        print(f"Failed IDs saved to: {failed_file}")
    
    return total_fetched, all_failed_ids


async def main():
    """Main verification and completion process."""
    print("=== Verify and Complete Citation Extraction ===")
    print("This script will verify your existing extracted data and fetch any missing documents.")
    print()
    
    # Step 1: Get existing records file
    existing_file = get_existing_file()
    
    # Step 2: Get synergy dataset
    dataset_name = prompt_synergy_dataset_selection()
    
    print(f"\n=== Analysis Configuration ===")
    print(f"Existing records file: {existing_file}")
    print(f"Synergy dataset: {dataset_name}")
    
    # Step 3: Load existing records
    try:
        existing_ids = load_existing_records(existing_file)
    except Exception as e:
        print(f"Error loading existing records: {e}")
        return
    
    # Step 4: Get expected IDs from synergy dataset
    try:
        dataset_ids, expected_ids = get_expected_ids_from_synergy(dataset_name)
    except Exception as e:
        print(f"Error loading synergy dataset: {e}")
        return
    
    # Step 5: Compare and find missing
    external_refs = expected_ids - dataset_ids  # Only external references
    missing_ids = external_refs - existing_ids
    
    print(f"\n=== Data Analysis ===")
    print(f"Documents in synergy dataset: {len(dataset_ids):,}")
    print(f"External references found: {len(external_refs):,}")
    print(f"Already extracted: {len(existing_ids):,}")
    print(f"Missing documents: {len(missing_ids):,}")
    
    if not missing_ids:
        print("✅ No missing documents! Your extraction is complete.")
        return
    
    print(f"\n📊 Completion Status: {len(existing_ids)}/{len(external_refs)} ({100*len(existing_ids)/len(external_refs):.1f}%)")
    
    # Step 6: Confirm fetching missing documents
    response = input(f"\nFetch {len(missing_ids)} missing documents? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Step 7: Setup temp directory
    temp_dir = Path('temp_records') / dataset_name.replace('-', '_').replace(' ', '_').lower()
    
    # Step 8: Fetch missing documents
    try:
        fetched_count, failed_ids = await fetch_missing_documents(missing_ids, temp_dir)
        
        if fetched_count == 0:
            print("No documents were successfully fetched.")
            return
        
        # Step 9: Merge existing and new data
        print(f"\n=== Merging Data ===")
        
        # Load existing data
        existing_df = pd.read_csv(existing_file)
        print(f"Loaded {len(existing_df)} existing records")
        
        # Load new data from temp files
        new_data = []
        for json_file in temp_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    work_data = json.load(f)
                row = prepare_work_data_for_csv(work_data)
                new_data.append(row)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        if new_data:
            new_df = pd.DataFrame(new_data)
            print(f"Loaded {len(new_df)} new records from temp files")
            
            # Combine dataframes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save complete file
            output_file = existing_file.parent / f"complete_{existing_file.name}"
            combined_df.to_csv(output_file, index=False)
            
            print(f"\n✅ Complete dataset saved to: {output_file}")
            print(f"Total records: {len(combined_df):,}")
            print(f"New records added: {len(new_df):,}")
            
            if failed_ids:
                print(f"⚠️  {len(failed_ids)} documents could not be fetched")
                success_rate = (len(external_refs) - len(failed_ids)) / len(external_refs) * 100
                print(f"Overall completion rate: {success_rate:.1f}%")
            else:
                print("🎉 100% completion rate achieved!")
                
        else:
            print("No new data to merge.")
            
    except Exception as e:
        print(f"Error during fetching: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 