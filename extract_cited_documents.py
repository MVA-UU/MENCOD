#!/usr/bin/env python3
"""
Extract Cited Documents from OpenAlex API

This script expands our datasets by extracting documents cited by papers in our 
existing datasets through the OpenAlex API. It respects rate limiting and saves
the results for later preprocessing.

Usage:
    python extract_cited_documents.py

The script will:
1. Show available datasets from synergy-dataset library
2. Load the selected synergy dataset using the synergy-dataset package
3. Extract all referenced works from papers in that dataset
4. Fetch detailed information for cited documents not already in our dataset
5. Save results to data/external_data/{dataset_name}.csv
"""

import asyncio
import aiohttp
import pandas as pd
import json
import os
import sys
import time
from typing import Set, List, Dict, Any, Optional


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
        """
        Initialize OpenAlex client.
        
        Args:
            max_concurrent_requests: Maximum concurrent requests (OpenAlex limit is 10)
        """
        self.base_url = "https://api.openalex.org"
        self.max_concurrent = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        
        # Add polite headers as recommended by OpenAlex
        self.headers = {
            'User-Agent': 'mailto:your-email@domain.com',  # Replace with your email
            'Accept': 'application/json'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_work_details(self, openalex_id: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a single work with robust retry logic.
        
        Args:
            openalex_id: OpenAlex ID of the work
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with work details or None if permanently failed
        """
        async with self.semaphore:  # Rate limiting
            # Clean the OpenAlex ID - remove any URL prefix
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
                            # Rate limited - use exponential backoff
                            wait_time = min(2 ** attempt, 60)  # Max 60 seconds
                            print(f"Rate limited for {clean_id} (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status == 404:
                            # Document not found - this is permanent, don't retry
                            print(f"Document {clean_id} not found (404)")
                            return None
                        elif response.status >= 500:
                            # Server error - temporary, should retry
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
            
            # Small delay to be polite to the API
            await asyncio.sleep(0.1)
            
        # If we get here, all retries failed
        print(f"PERMANENT FAILURE: Could not fetch {clean_id} after {max_retries + 1} attempts")
        return None
    
    async def get_works_batch_with_retry_queue(self, openalex_ids: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Get details for multiple works concurrently with a retry queue for failed requests.
        
        Args:
            openalex_ids: List of OpenAlex IDs
            
        Returns:
            Tuple of (successful_results, failed_ids)
        """
        tasks = [self.get_work_details(openalex_id) for openalex_id in openalex_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from failures
        successful_results = []
        failed_ids = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception for {openalex_ids[i]}: {result}")
                failed_ids.append(openalex_ids[i])
            elif result is not None:
                successful_results.append(result)
            else:
                # None result means permanent failure after retries
                failed_ids.append(openalex_ids[i])
        
        return successful_results, failed_ids


def prompt_synergy_dataset_selection() -> str:
    """
    Prompt user to select a dataset from available synergy datasets.
    
    Returns:
        Selected dataset name
    """
    try:
        datasets = []
        print("Loading available datasets...")
        for d in iter_datasets():
            datasets.append(d.name)
        
        if not datasets:
            raise ValueError("No datasets available")
            
    except Exception as e:
        print(f"Error getting available datasets: {e}")
        print("Trying some common dataset names...")
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


def load_synergy_dataset_with_package(dataset_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the synergy dataset using the synergy-dataset package.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Appenzeller-Herzog_2019')
        
    Returns:
        Dictionary mapping OpenAlex IDs to work data
    """
    print(f"Loading synergy dataset: {dataset_name}")
    try:
        dataset = Dataset(dataset_name)
        
        # Get the data as a dictionary with the specific fields we need
        data_dict = dataset.to_dict(["title", "referenced_works"])
        print(f"Loaded {len(data_dict)} documents from synergy dataset")
        
        # Show available fields for the first document
        if data_dict:
            first_key = next(iter(data_dict))
            first_doc = data_dict[first_key]
            print(f"Available fields: {list(first_doc.keys())}")
            
            # Check if we have the required fields
            has_refs = 'referenced_works' in first_doc
            print(f"Has 'referenced_works' field: {has_refs}")
            
            # Show sample of first document
            print(f"\nSample document data:")
            print(f"  OpenAlex ID: {first_key}")
            print(f"  Title: {first_doc.get('title', '')[:100]}...")
            ref_works = first_doc.get('referenced_works', [])
            print(f"  Referenced works count: {len(ref_works)}")
            if ref_works:
                print(f"  First few references: {ref_works[:3]}")
        
        return data_dict
        
    except Exception as e:
        print(f"Error loading synergy dataset '{dataset_name}': {e}")
        print("Available datasets:")
        try:
            datasets = [d.name for d in iter_datasets()]
            print(f"Available: {datasets}")
        except:
            print("Could not retrieve available datasets")
        raise


def extract_openalex_data_from_synergy(data_dict: Dict[str, Dict[str, Any]]) -> tuple[Set[str], Set[str]]:
    """
    Extract OpenAlex IDs and referenced works from synergy dataset.
    
    Args:
        data_dict: Dictionary of synergy dataset data
        
    Returns:
        Tuple of (existing_ids, all_referenced_works)
    """
    existing_ids = set()
    all_referenced_works = set()
    
    for openalex_url, doc_data in data_dict.items():
        # The key is already the OpenAlex URL, extract the clean ID
        clean_id = openalex_url.replace('https://openalex.org/', '')
        existing_ids.add(clean_id)
        
        # Extract referenced works
        referenced_works = doc_data.get('referenced_works', [])
        if referenced_works and isinstance(referenced_works, list):
            for ref_work in referenced_works:
                if isinstance(ref_work, str):
                    # Clean the referenced work ID
                    clean_ref_id = ref_work.replace('https://openalex.org/', '')
                    all_referenced_works.add(clean_ref_id)
    
    print(f"Found {len(existing_ids)} existing OpenAlex IDs")
    print(f"Found {len(all_referenced_works)} total referenced works")
    
    return existing_ids, all_referenced_works


def prepare_work_data_for_csv(work_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant fields from OpenAlex work data for CSV storage.
    
    Args:
        work_data: Raw OpenAlex work data
        
    Returns:
        Dictionary with cleaned data for CSV
    """
    # Extract basic information
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
        for authorship in work_data['authorships'][:5]:  # Limit to first 5 authors
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
    
    # Extract concepts (topics/subjects)
    concepts = []
    if 'concepts' in work_data and work_data['concepts']:
        for concept in work_data['concepts'][:10]:  # Limit to top 10 concepts
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
    
    # Extract mesh terms if available
    mesh_terms = []
    if 'mesh' in work_data and work_data['mesh']:
        for mesh in work_data['mesh'][:10]:  # Limit to 10 mesh terms
            if mesh.get('descriptor_name'):
                mesh_terms.append(mesh['descriptor_name'])
    row['mesh_terms'] = '; '.join(mesh_terms) if mesh_terms else ''
    
    return row


async def extract_cited_documents(dataset_name: str):
    """
    Main function to extract cited documents for a dataset.
    
    Args:
        dataset_name: Name of the synergy dataset to process
    """
    print(f"\n=== Extracting cited documents for {dataset_name} ===")
    
    # Load synergy dataset using the synergy-dataset package
    try:
        synergy_data = load_synergy_dataset_with_package(dataset_name)
    except Exception as e:
        print(f"Failed to load synergy dataset: {e}")
        return
    
    # Extract OpenAlex IDs and referenced works
    existing_ids, all_referenced_works = extract_openalex_data_from_synergy(synergy_data)
    
    if not existing_ids:
        print("ERROR: No OpenAlex IDs found in dataset. Cannot proceed.")
        return
    
    if not all_referenced_works:
        print("WARNING: No referenced works found in dataset.")
        return
    
    # Step 1: Filter out works that are already in our dataset
    new_works = all_referenced_works - existing_ids
    print(f"New works not in our dataset: {len(new_works)}")
    
    if not new_works:
        print("No new works to fetch. All referenced works are already in our dataset.")
        return
    
    # Step 2: Setup output directories
    output_dir = os.path.join('data', 'external_data')
    records_dir = os.path.join(output_dir, 'records', dataset_name.replace('-', '_').replace(' ', '_').lower())
    os.makedirs(records_dir, exist_ok=True)
    
    print(f"Records will be saved to: {records_dir}")
    
    # Step 3: Check for existing records (resume capability)
    existing_files = set()
    if os.path.exists(records_dir):
        for filename in os.listdir(records_dir):
            if filename.endswith('.json'):
                work_id = filename[:-5]  # Remove .json extension
                existing_files.add(work_id)
    
    # Filter out works we've already fetched
    new_works_to_fetch = new_works - existing_files
    already_fetched = len(new_works) - len(new_works_to_fetch)
    
    print(f"Already fetched: {already_fetched} works")
    print(f"Still need to fetch: {len(new_works_to_fetch)} works")
    
    if not new_works_to_fetch:
        print("All works have already been fetched! Proceeding to merge...")
    else:
        # Step 4: Fetch detailed information for new works
        print(f"\nStep 1: Fetching detailed information for {len(new_works_to_fetch)} new works...")
        
        new_works_list = list(new_works_to_fetch)
        batch_size = 50  # Process in batches to manage memory
        total_fetched = 0
        all_failed_ids = set()  # Track permanently failed IDs
        
        async with OpenAlexClient() as client:
            for i in range(0, len(new_works_list), batch_size):
                batch = new_works_list[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(new_works_list) + batch_size - 1)//batch_size
                
                print(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} works)")
                
                # Get detailed information for this batch
                works_data, failed_ids = await client.get_works_batch_with_retry_queue(batch)
                
                # Save each work immediately
                batch_saved = 0
                for work_data in works_data:
                    if work_data:
                        try:
                            work_id = work_data.get('id', '').replace('https://openalex.org/', '')
                            if work_id:
                                # Save as JSON file
                                record_file = os.path.join(records_dir, f"{work_id}.json")
                                with open(record_file, 'w', encoding='utf-8') as f:
                                    json.dump(work_data, f, indent=2, ensure_ascii=False)
                                batch_saved += 1
                                total_fetched += 1
                        except Exception as e:
                            print(f"Error saving work data: {e}")
                            continue
                
                # Track failed IDs
                if failed_ids:
                    all_failed_ids.update(failed_ids)
                    print(f"Failed to fetch {len(failed_ids)} works in this batch")
                
                print(f"Saved {batch_saved} works from this batch (Total: {total_fetched + already_fetched}/{len(new_works)})")
                
                # Small delay between batches
                await asyncio.sleep(0.5)
        
        print(f"Finished fetching! Total works saved: {total_fetched}")
        
        # Handle permanently failed IDs
        if all_failed_ids:
            print(f"\n⚠️  WARNING: {len(all_failed_ids)} documents could not be fetched after multiple retries:")
            
            # Save failed IDs to a file for manual inspection
            failed_file = os.path.join(records_dir, 'failed_ids.txt')
            with open(failed_file, 'w') as f:
                for failed_id in sorted(all_failed_ids):
                    f.write(f"{failed_id}\n")
            
            print(f"Failed IDs saved to: {failed_file}")
            print(f"Success rate: {total_fetched}/{total_fetched + len(all_failed_ids)} ({100 * total_fetched / (total_fetched + len(all_failed_ids)):.1f}%)")
            
            # Ask user if they want to continue despite failures
            response = input("\nSome documents failed to fetch. Continue with merging available data? (y/n): ").lower().strip()
            if response != 'y':
                print("Stopping. You can re-run the script later to retry failed documents.")
                return
        else:
            print("✅ All documents fetched successfully!")
    
    # Step 5: Merge all individual records into CSV
    print(f"\nStep 2: Merging all records into CSV...")
    
    csv_data = []
    processed_count = 0
    error_count = 0
    
    # Read all JSON files in the records directory
    for filename in os.listdir(records_dir):
        if filename.endswith('.json'):
            try:
                record_file = os.path.join(records_dir, filename)
                with open(record_file, 'r', encoding='utf-8') as f:
                    work_data = json.load(f)
                
                row = prepare_work_data_for_csv(work_data)
                csv_data.append(row)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} records...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                error_count += 1
                continue
    
    print(f"Successfully processed {processed_count} records")
    if error_count > 0:
        print(f"Errors in {error_count} records")
    
    # Step 6: Save final CSV
    if csv_data:
        simple_name = dataset_name.replace('-', '_').replace(' ', '_').lower()
        output_file = os.path.join(output_dir, f'{simple_name}.csv')
        
        df_output = pd.DataFrame(csv_data)
        df_output.to_csv(output_file, index=False)
        
        print(f"\nSaved {len(csv_data)} cited documents to: {output_file}")
        print(f"Individual records saved in: {records_dir}")
        print(f"Columns: {list(df_output.columns)}")
        print(f"Sample data:")
        print(df_output[['title', 'publication_year', 'cited_by_count']].head())
        
        # Show some statistics
        total_refs = df_output['referenced_works_count'].sum()
        avg_refs = df_output['referenced_works_count'].mean()
        print(f"\nStatistics:")
        print(f"  Total cited documents extracted: {len(csv_data)}")
        print(f"  Total references in extracted documents: {total_refs:,}")
        print(f"  Average references per document: {avg_refs:.1f}")
        
    else:
        print("No valid work data to save!")


async def main():
    """Main entry point."""
    print("=== Extract Cited Documents from OpenAlex API ===")
    print("This script will expand your dataset by extracting documents cited by papers in your existing dataset.")
    print("Rate limiting: Maximum 10 concurrent requests to respect OpenAlex API limits.")
    print("Uses synergy-dataset package to access full OpenAlex Work objects.")
    print("Individual records are saved as JSON files for fault tolerance and resumability.")
    
    # Select dataset from synergy-dataset library
    dataset_name = prompt_synergy_dataset_selection()
    print(f"Selected dataset: {dataset_name}")
    
    # Check if there are existing records
    records_dir = os.path.join('data', 'external_data', 'records', dataset_name.replace('-', '_').replace(' ', '_').lower())
    existing_records = 0
    if os.path.exists(records_dir):
        existing_records = len([f for f in os.listdir(records_dir) if f.endswith('.json')])
    
    if existing_records > 0:
        print(f"\nFound {existing_records} existing records for this dataset.")
        print("The script will resume from where it left off and only fetch missing records.")
    
    # Confirm before proceeding
    print(f"\nThis will extract all documents cited by papers in the {dataset_name} dataset.")
    print("Individual records will be saved as JSON files for fault tolerance.")
    print("This may take several minutes depending on the dataset size.")
    
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run the extraction
    try:
        await extract_cited_documents(dataset_name)
        print("\n=== Extraction completed successfully! ===")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nProgress has been saved to individual JSON files.")
        print(f"You can run the script again to resume from where it left off.")


if __name__ == "__main__":
    asyncio.run(main()) 