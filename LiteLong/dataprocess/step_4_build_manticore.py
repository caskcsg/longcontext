import json
import time
import sys
import os
import glob
import random
from multiprocessing import Pool, cpu_count, Process
import argparse
import numpy as np

import requests
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq

def process_batch(batch_data, index_name, num_shards):
    """Batch process a group of documents for indexing"""
    if not batch_data:
        return {"indexed": 0, "failed": 0}
    
    ndjson = ""
    count = 0
    
    # Select a random shard for load balancing
    shard_index = f"{index_name}{random.randint(0, num_shards-1)}"
    
    for doc_id, content in batch_data:
        index_doc = {
            "insert": {
                "index": shard_index,
                "_id": doc_id,
                "doc": {
                    "content": content,
                    "fw_id": doc_id,
                    # You can add more useful metadata here
                },
            }
        }
        ndjson += json.dumps(index_doc) + "\n"
        count += 1
    
    response = None
    retries = 0
    max_retries = 5
    
    while response is None and retries < max_retries:
        try:
            response = requests.post(
                "http://127.0.0.1:9308/bulk",
                headers={"Content-Type": "application/x-ndjson"},
                data=ndjson,
                timeout=36000
            )
            if response.status_code != 200:
                print(f"Batch indexing error: {response.status_code}, {response.text}", file=sys.stderr)
                retries += 1
                response = None
                time.sleep(2)
            else:
                return {"indexed": count, "failed": 0}
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}", file=sys.stderr)
            retries += 1
            time.sleep(2)
            response = None
    
    return {"indexed": 0, "failed": count}

def load_documents(file_paths, batch_size=100):
    """Load documents and yield in batches, supporting JSON and Parquet formats"""
    current_batch = []
    
    for file_path in file_paths:
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.json':
                # Handle JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                content = doc.get("text", "")
                
                # Filter out content that is too short
                if len(content.strip()) < 20:
                    continue
                    
                current_batch.append((doc_id, content))
            elif file_ext == '.jsonl':
                import json
                data = json.load(open(file_path, "r"))
                for key, val in data.items():
                    doc_id = key
                    content = val
                    current_batch.append((doc_id, content))
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []

                        
            
            elif file_ext == '.parquet':
                # Handle Parquet file
                try:
                    # Read Parquet file
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    
                    # Assume the Parquet file has a 'text' or 'content' column
                    if 'text' in df.columns:
                        content_column = 'text'
                    elif 'content' in df.columns:
                        content_column = 'content'
                    else:
                        print(f"Warning: {file_path} does not have a text column, skipping", file=sys.stderr)
                        continue
                    
                    # Process each row
                    for idx, row in df.iterrows():
                        content = str(row[content_column])
                        
                        # Filter out content that is too short
                        if len(content.strip()) < 20:
                            continue
                        
                        
                        # Create a unique document ID (filename + row number)
                        doc_id = f"{os.path.splitext(os.path.basename(file_path))[0]}_{idx}"
                        
                        current_batch.append((doc_id, content))
                        
                        # Yield batch when it reaches the specified size
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                
                except Exception as e:
                    print(f"Error processing Parquet file {file_path}: {e}", file=sys.stderr)
                
                # For Parquet, batches are already handled in the loop, so skip the outer batch check
                continue
            
            # Yield batch when it reaches the specified size
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
    
    # Yield any remaining documents
    if current_batch:
        yield current_batch

def create_and_setup_index(index_name, num_shards, language='en'):
    """Create and set up an index with multiple shards"""
    sql_url = "http://127.0.0.1:9308/sql?mode=raw"
    
    print(f"Initializing index {index_name}...", file=sys.stderr)
    
    # Create distributed index schema
    for i in range(num_shards):
        # Try to unfreeze table (if exists)
        try:
            response = requests.post(
                sql_url, 
                data={"query": f"UNFREEZE {index_name}{i}"},
                timeout=36000
            )
            print(f"Unfreeze table {index_name}{i}: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to unfreeze table {index_name}{i}: {e}", file=sys.stderr)

        # Drop existing table first
        try:
            response = requests.post(
                sql_url, data={"query": f"drop table if exists {index_name}{i}"},
                timeout=36000
            )
            print(f"Dropped table {index_name}{i}: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to drop table {index_name}{i}: {e}", file=sys.stderr)

        # Use a more complete table structure
        try:
            # Choose table config based on language
            if language == 'zh':
                # Chinese config: enable Chinese tokenization
                local_query = f"create table {index_name}{i}(content text, fw_id string) charset_table='cjk' morphology='icu_chinese'"
            else:
                # English default config
                local_query = f"create table {index_name}{i}(content text, fw_id string) charset_table='non_cjk' stopwords='en' morphology='stem_en'"
            
            response = requests.post(sql_url, data={"query": local_query}, timeout=36000)
            
            # Check if successful
            if "error" in response.text.lower():
                print(f"Warning: Creating table {index_name}{i} may have failed: {response.text.strip()}", file=sys.stderr)
            else:
                print(f"Created table {index_name}{i}: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to create table {index_name}{i}: {e}", file=sys.stderr)
        
    
    # Drop distributed table if exists
    try:
        response = requests.post(sql_url, data={"query": f"drop table if exists {index_name}"}, timeout=36000)
        print(f"Dropped existing distributed table: {response.text.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to drop distributed table {index_name}: {e}", file=sys.stderr)
    
    # Create distributed table
    distributed_query = f"create table {index_name} type='distributed'"
    for i in range(num_shards):
        distributed_query += f" local='{index_name}{i}'"
    
    try:
        response = requests.post(sql_url, data={"query": distributed_query}, timeout=36000)
        
        # Check if successful
        if "error" in response.text.lower():
            print(f"Warning: Creating distributed table may have failed: {response.text.strip()}", file=sys.stderr)
        else:
            print(f"Created distributed table: {response.text.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to create distributed table {index_name}: {e}", file=sys.stderr)

def process_subset(subset_name, json_files, num_shards, language='en'):
    """Process index creation for a subset"""
    # Create index
    create_and_setup_index(subset_name, num_shards, language)
    
    total_files = len(json_files)
    print(f"{subset_name}: Found {total_files} JSON files", file=sys.stderr)
    
    # Use multiprocessing for indexing
    num_processes = min(cpu_count(), num_shards)
    batch_size = 1000
    
    print(f"{subset_name}: Using {num_processes} processes for indexing, batch size: {batch_size}", file=sys.stderr)
    
    # Stats variables
    total_indexed = 0
    total_failed = 0
    
    with Pool(num_processes) as pool:
        active_tasks = []
        
        # Use tqdm to show progress
        with tqdm(total=total_files, desc=f"{subset_name} indexing progress") as pbar:
            for batch in load_documents(json_files, batch_size):
                # Limit number of active tasks to avoid high memory usage
                while len(active_tasks) >= num_processes * 2:
                    # Check completed tasks and update progress
                    still_active = []
                    for batch_size, async_result in active_tasks:
                        if async_result.ready():
                            result = async_result.get()
                            total_indexed += result["indexed"]
                            total_failed += result["failed"]
                            pbar.update(batch_size)
                            pbar.set_postfix(indexed=total_indexed, failed=total_failed)
                        else:
                            still_active.append((batch_size, async_result))
                    active_tasks = still_active
                    
                    # If no tasks completed, wait briefly
                    if len(active_tasks) >= num_processes * 2:
                        time.sleep(0.1)
                
                # Submit new task
                async_result = pool.apply_async(process_batch, (batch, subset_name, num_shards))
                active_tasks.append((len(batch), async_result))
            
            # Wait for all remaining tasks to finish
            while active_tasks:
                still_active = []
                for batch_size, async_result in active_tasks:
                    if async_result.ready():
                        result = async_result.get()
                        total_indexed += result["indexed"]
                        total_failed += result["failed"]
                        pbar.update(batch_size)
                        pbar.set_postfix(indexed=total_indexed, failed=total_failed)
                    else:
                        still_active.append((batch_size, async_result))
                
                active_tasks = still_active
                if active_tasks:
                    time.sleep(0.1)
    
    print(f"{subset_name} indexing complete: success {total_indexed}, failed {total_failed}", file=sys.stderr)
    
    # Wait for all data to be written
    time.sleep(10)
    
    # Optimize each shard
    sql_url = "http://127.0.0.1:9308/sql?mode=raw"
    for i in range(num_shards):
        print(f"Optimizing table {subset_name}{i}...", file=sys.stderr)
        
        # Flush
        try:
            response = requests.post(
                sql_url,
                data={"query": f"FLUSH TABLE {subset_name}{i}"},
                timeout=36000,
            )
            print(f"FLUSH: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"FLUSH failed: {e}", file=sys.stderr)
        
        # Optimize
        try:
            response = requests.post(
                sql_url,
                data={"query": f"OPTIMIZE TABLE {subset_name}{i} OPTION cutoff=16, sync=1"},
                timeout=36000,
            )
            print(f"OPTIMIZE: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"OPTIMIZE failed: {e}", file=sys.stderr)
        
        # Freeze index to improve search performance
        try:
            response = requests.post(
                sql_url,
                data={"query": f"FREEZE {subset_name}{i}"},
                timeout=36000,
            )
            print(f"FREEZE: {response.text.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"FREEZE failed: {e}", file=sys.stderr)
    
    # Test search
    test_queries = ["artificial intelligence", "machine learning", "deep learning"]
    for query in test_queries:
        response = requests.post(
            "http://127.0.0.1:9308/search",
            data=json.dumps({
                "index": subset_name,
                "query": {"match": {"*": query}},
                "limit": 5
            }),
            headers={"Content-Type": "application/json"},
            timeout=36000
        )
        print(f"{subset_name} test search '{query}':", file=sys.stderr)
        try:
            results = json.loads(response.text)
            hits = results.get("hits", {}).get("total", 0)
            print(f"Found {hits} results", file=sys.stderr)
        except:
            print(f"Search response: {response.text[:200]}...", file=sys.stderr)

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Index documents into Manticore Search')
    parser.add_argument('--corpus', type=str, default='pdfdrive', help='Corpus name for index')
    parser.add_argument('--num_shards', type=int, default=1, help='Number of shards')
    parser.add_argument('--num_subsets', type=int, default=1, help='Number of subsets')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'], help='Language for tokenization (en=English, zh=Chinese)')
    args = parser.parse_args()
    
    # Create local data directory
    data_dir = "./manticore_data"
    os.makedirs(data_dir, exist_ok=True)
    
    if args.corpus in ["fineweb_edu_dedup", "cosmopedia_v2",]:
        # zlib processing logic
        if args.corpus in ["fineweb_edu_dedup", "cosmopedia_v2"]:
            base_path = f"xxx/smollm-corpus/{args.corpus}/"
        parquet_files = glob.glob(os.path.join(base_path, "*.parquet"))
        print(f"Found {len(parquet_files)} Parquet files in {base_path}", file=sys.stderr)
        # import ipdb; ipdb.set_trace()
        # Split Parquet files into num_subsets subsets
        subsets = np.array_split(parquet_files, args.num_subsets)
        
        # Create a process for each subset
        processes = []
        for i, subset in enumerate(subsets):
            subset_name = f"{args.corpus}_subset{i}"
            
            print(f"Subset {i}: {len(subset)} Parquet files", file=sys.stderr)
            
            # Start processing process
            p = Process(target=process_subset, args=(subset_name, subset, args.num_shards, args.language))
            p.start()
            processes.append(p)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        print(f"All {args.corpus} subsets processed", file=sys.stderr)


if __name__ == "__main__":
    main()
