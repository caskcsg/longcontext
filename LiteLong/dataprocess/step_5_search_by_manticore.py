import argparse
import json
import sys
import time
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial

import requests
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str
    )
    parser.add_argument("--n_pages", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="search_results",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_subsets", type=int, default=1)
    parser.add_argument("--corpus", type=str)
    return parser.parse_args()



args = get_args()

# 创建输出目录
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# 读取查询文件
with open(args.input_file, 'r', encoding='utf-8') as f:
    queries = f.read().splitlines()

shard_queries = queries

args.output_dir = Path(args.output_dir) / args.corpus


def run_query(query, n_pages, index_name):
    while True:
        try:
            max_pages = 4_000
            response = requests.post(
                "http://127.0.0.1:9308/search",
                data=json.dumps(
                    {
                        "index": index_name,
                        "size": n_pages,
                        "query": query,
                        "max_matches": max_pages,
                    }
                ),
                timeout=1000,
            )
            if response.status_code != 200:
                print(response.text, file=sys.stderr)
                time.sleep(5)
                continue
            else:
                hits = response.json()["hits"]["hits"]
                return hits
        except requests.exceptions.ConnectionError as e:
            print(e, file=sys.stderr)
            time.sleep(5)
            continue


def process_query(query, args, subset_idx):
    # 创建查询专属文件夹
    query_dir = os.path.join(args.output_dir, query.replace(' ', '_')[:50])
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    
    # 准备查询字符串
    clean_query = query
    for c in ["!", '"', "$", "'", "(", ")", "/", "<", "@", "\\", "^", "|", "~"]:
        clean_query = clean_query.replace(c, " ")
    
    if args.num_subsets > 1:
        if args.num_shards == 1:
            index_name = f"{args.corpus}_subset{subset_idx}0"
        else:
            index_name = f"{args.corpus}_subset{subset_idx}"
    else:
        index_name = args.corpus
    
    # 运行查询
    try:
        hits = run_query({"query_string": clean_query}, args.n_pages, index_name)
        print(f"Got {len(hits)} hits for query: {query} from {index_name}", file=sys.stderr)
        
        result_dir = os.path.join(query_dir, f"subset_{subset_idx}")
        os.makedirs(result_dir, exist_ok=True)
        
        for i, hit in enumerate(hits):
            result_file = os.path.join(result_dir, f"result_{i}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(hit, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(hits)} results for query: {query} from {index_name}")
        return hits
    except Exception as e:
        print(f"Error with index {index_name}: {e}", file=sys.stderr)
        return []


def process_queries(queries, args):

    num_processes = min(mp.cpu_count(), len(queries))
    pool = mp.Pool(processes=num_processes)
    

    process_query_partial = partial(process_all_subsets, args=args)
    
    try:

        pool.map(process_query_partial, queries)
    finally:
        pool.close()
        pool.join()

def process_all_subsets(query, args):
    try:
        all_hits = []

        for subset_idx in range(args.num_subsets):
            hits = process_query(query, args, subset_idx)
            all_hits.extend(hits)
        

        if args.num_subsets > 1:
            merged_dir = os.path.join(args.output_dir, query.replace(' ', '_')[:50], "merged")
            os.makedirs(merged_dir, exist_ok=True)
            
            for i, hit in enumerate(all_hits):
                result_file = os.path.join(merged_dir, f"result_{i}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(hit, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(all_hits)} merged results for query: {query}")
            
    except Exception as e:
        print(f"Error processing query '{query}': {e}", file=sys.stderr)

if __name__ == '__main__':

    process_queries(shard_queries, args)
