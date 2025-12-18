import os
import json
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import sys
import argparse


length_list = {
    '64k': 393216,
    '128k': 786432,
    '512k': 3145728,
}


def process_theme(theme, length='512k', data_length=4, method='shuffle'):
    try:
        file_dirs = []
        file_dir = os.path.join(root_dir, theme, "merged")
        if not os.path.exists(file_dir):
            theme_dir = os.path.join(root_dir, theme)
            for subdir in os.listdir(theme_dir):
                potential_dir = os.path.join(theme_dir, subdir, "merged")
                if os.path.exists(potential_dir):
                    file_dir = potential_dir
                    file_dirs.append(file_dir)
        else:
            file_dirs.append(file_dir)
        
        result = []
        for file_dir in file_dirs:
            files = os.listdir(file_dir)
            
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]) if x.startswith('result_') and x.split('_')[1].split('.')[0].isdigit() else float('inf'))
            file_paths = [os.path.join(file_dir, f) for f in files]
            
            theme_datas = []
            theme_scores = []
            for file_path in file_paths:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    theme_datas.append(data['_source']['content'])
                    theme_scores.append(data['_score'])
            import pdb
            theme_datas = list(set(theme_datas))
            if method != 'shuffle':
                theme_datas = [x for _, x in sorted(zip(theme_scores, theme_datas), key=lambda pair: pair[0], reverse=True)]
                
            data_list = process_data(theme_datas, length, data_length, method)
            result.extend(data_list)
        return {'theme': theme, 'text': result, 'success': True}
    except Exception as e:
        return {'theme': theme, 'success': False, 'error': str(e)}

def process_data(theme_datas, length='512k', data_length=4, method='shuffle'):
    result_list = []
    current_chunk = ""
    total_len = 0
    index_list = list(range(len(theme_datas)))

    chunk_size = length_list[length]
    if method == 'reverse':
        index_list.reverse()
    if method == 'shuffle':
        import random
        random.shuffle(index_list)
    for i in index_list:
        d = theme_datas[i]
        if total_len > chunk_size:
            if current_chunk: 
                result_list.append(current_chunk)
                current_chunk = ""
                total_len = 0
        else:
            if len(d) > chunk_size:
                result_list.append(d[:chunk_size])
            else:
                current_chunk = current_chunk + d + '\n\n'
                total_len += len(d)
        if len(result_list) >= data_length:
            break
    
    if len(current_chunk) > chunk_size and len(result_list) < data_length:
        result_list.append(current_chunk)
    if len(current_chunk) and len(result_list) == 0:
        result_list.append(current_chunk)
    
    return result_list[:data_length]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge JSONL files from multiple folders')
    parser.add_argument('--split_id', type=int, help='Split ID to filter files')
    parser.add_argument('--length', type=str, help='Length to filter files')
    parser.add_argument('--root_dir', type=str, help='Length to filter files')
    parser.add_argument('--dataset_length', type=int, default=4)
    parser.add_argument('--method', type=str, default='shuffle')
    
    args = parser.parse_args()
    
    split_id = args.split_id
    root_dir = args.root_dir
    
    with open(f'sub_{split_id}/themes.txt', 'r') as f:
        themes = [line.strip() for line in f]
    
    name = args.root_dir.split('/')[-2]
    output_file = f'sub_{split_id}/{name}.jsonl'
    log_file = f'sub_{split_id}/{name}.log'
    
    processed_themes = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_themes = set(line.strip() for line in f)
    
    themes_to_process = [theme for theme in themes if theme not in processed_themes]

    final_theme = []
    for theme in themes_to_process:
        if length_flirt(theme):
            final_theme.append(theme)
    print(len(final_theme))
    num_workers = 20 
    
    results = []
    failed = []
    themes_to_process = final_theme

    file_mode = 'a+' if processed_themes else 'w+'
    current_line = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_theme, theme, args.length, args.dataset_length, args.method) for theme in themes_to_process]
        
        with open(output_file, file_mode) as w_file, open(log_file, 'a+') as log:
            for future in tqdm(futures, total=len(themes_to_process), desc="process theme"):
                result = future.result()
                if result['success']:
                    count = 0
                    for text in result['text']:
                        w_file.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        w_file.flush() 
                        count += 1
                    current_line += len(result['text'])
                
                    theme_name = result['theme']
                    results.append(theme_name)
                    log.write(f"{theme_name}\n")
                    log.flush() 
                else:
                    failed.append(f"{result['theme']}: {result.get('error', 'error')}")
    
    if failed:
        print("themes that failed:")
        for error in failed:    
            print(error)
    
    print(f'success: {len(results)}')
    print(f'total lines: {current_line}')

