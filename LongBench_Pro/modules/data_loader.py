import os
import json
import logging
from datasets import Dataset, load_dataset
from typing import List, Dict, Set, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """data loader"""
    
    def __init__(self, dataset_path: str = 'dataset/longbench_pro.json', bon_num: int = 1, total_shards: int = 1, shard_id: int = 1, seed: int = 42) -> None:
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.DataLoader")
        self.inference_dataset, self.inference_samples_num, self.evaluation_samples_num = self.load_dataset_with_bon_shard(dataset_path, bon_num, total_shards, shard_id, seed)
    
    def load_dataset_with_bon_shard(self, dataset_path: str, bon_num: int = 3, total_shards: int = 1, shard_id: int = 1, seed: int = 42) -> Tuple[List[Dict[str, Any]], int, int]:
        """load dataset with bon and shard"""
        self.logger.info("Starting to load dataset")
        try:
            try:
                original_dataset: List[Dict[str, Any]] = json.load(open(dataset_path, 'r', encoding='utf-8'))
            except:
                self.logger.info(f"Dataset file not found: {dataset_path}, loading from HuggingFace (caskcsg/LongBench_Pro)")
                original_dataset: Dataset = load_dataset("caskcsg/LongBench_Pro", split='test')
            bon_dataset: List[Dict[str, Any]] = []
            # repeat the dataset bon_num times, to get the best of bon_num results
            for i in range(bon_num):
                for item in original_dataset:
                    tmp_data = {
                        "bon_idx": i + 1, # use for filtering data
                        "id": item["id"],
                        "context": item["context"],
                        "language": item["language"],
                        "token_length": item["token_length"],
                        "primary_task": item["primary_task"],
                        "secondary_task": item["secondary_task"],
                        "contextual_requirement": item["contextual_requirement"],
                        "question_nonthinking": item["question_nonthinking"],
                        "question_thinking": item["question_thinking"],
                        "answer": item["answer"],
                        "difficulty": item["difficulty"]
                    }
                    bon_dataset.append(tmp_data)
            # shuffle and shard the dataset
            shuffle_dataset: Dataset = Dataset.from_list(bon_dataset)
            shuffle_dataset = shuffle_dataset.shuffle(seed=seed)
            shard_dataset = shuffle_dataset.shard(
                num_shards=total_shards, index=shard_id - 1
            )
            inference_dataset: List[Dict[str, Any]] = shard_dataset.to_list()
            inference_samples_num: int = len(inference_dataset)
            evaluation_samples_num: int = len(original_dataset) * bon_num
            self.logger.info("Dataset loaded successfully!")
            self.logger.info(f"Number of inference samples: [Original-{len(original_dataset)} * BoN-{bon_num} / Shard-{total_shards}] = {inference_samples_num}")
            self.logger.info(f"Number of evaluation samples: [Original-{len(original_dataset)} * BoN-{bon_num}] = {evaluation_samples_num}")
        except FileNotFoundError:
            self.logger.error(f"Dataset file not found: {dataset_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Dataset file format error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unknown error occurred while loading dataset: {e}")
            raise
        return inference_dataset, inference_samples_num, evaluation_samples_num
    
    def get_cached_data(self, output_file: str) -> Dict[str, Any]:
        """get cached data id list"""
        has_data: Dict[str, Any] = {}
        if os.path.exists(output_file):
            self.logger.info(f"Found cached file: {output_file}")
            try:
                with open(output_file, encoding='utf-8') as f:
                    line_count: int = 0
                    empty_prediction_count: int = 0
                    for line in f:
                        line_count += 1
                        try:
                            item: Dict[str, Any] = json.loads(line)
                            # check prediction is empty
                            if not item.get("prediction") or item.get("prediction").strip() == "" or item.get("prediction") is None:
                                empty_prediction_count += 1
                                continue
                            has_data[f"{item['id']}-{item['bon_idx']}"] = item
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Line {line_count} JSON format error, skipping: {e}")
                            continue
                
                self.logger.info(f"Cached data processing completed - Total lines: {line_count}, Valid data: {len(has_data)}, Empty prediction: {empty_prediction_count}")
            except Exception as e:
                self.logger.error(f"Error occurred while reading cached file: {e}")
                raise
        else:
            self.logger.warning(f"Cached file does not exist: {output_file}")
        
        return has_data
    
    def save_has_data(self, has_data: Dict[str, Any], output_file: str) -> None:
        """save has data to output_file"""
        try:
            backup_file: str = output_file + '.backup'
            if os.path.exists(output_file):
                os.rename(output_file, backup_file)
                self.logger.info(f"Backup file created: {backup_file}")
            else:
                self.logger.info(f"Output file does not exist, will create new file: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                for item in has_data.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"Valid data saved to: {output_file}, total {len(has_data)} items")
        except Exception as e:
            self.logger.error(f"Error occurred while saving data: {e}")
            raise
    
    def filter_new_data(self, output_file: str) -> List[Dict[str, Any]]:
        """filter out unprocessed data and data with empty prediction"""
        self.logger.info("Starting to filter data that needs processing")
        has_data: Dict[str, Any] = self.get_cached_data(output_file)
        self.save_has_data(has_data, output_file)
        data: List[Dict[str, Any]] = []
        for item in self.inference_dataset:
            if f"{item['id']}-{item['bon_idx']}" not in has_data:
                data.append(item)
        self.logger.info(f"Data filtering completed - Data to process: {len(data)} items")
        return data
    
    def split_data_for_multi_process(self, data: List[Dict[str, Any]], n_proc: int) -> List[List[Dict[str, Any]]]:
        """split data for multi process"""
        data_subsets: List[List[Dict[str, Any]]] = [data[i::n_proc] for i in range(n_proc)]
        return data_subsets

