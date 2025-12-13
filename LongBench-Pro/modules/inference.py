import time
import logging
from tqdm import tqdm
import torch.multiprocessing as mp
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceEngine:
    """inference engine"""

    def __init__(self, model_manager: Any, thinking_enabled: bool) -> None:
        self.model_manager = model_manager
        self.thinking_enabled = thinking_enabled

    def prepare_prompt(self, item: Dict[str, Any]) -> str:
        """prepare prompt"""
        context = item['context']
        if self.thinking_enabled:
            question = item['question_thinking']
        else:
            question = item['question_nonthinking']
        return f"{context}\n\n\n\n{question}"
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """process single item"""
        prompt = self.prepare_prompt(item)
        prediction, thinking = self.model_manager.query(prompt)
        if prediction is not None:
            prediction = prediction.strip()
        else:
            prediction = ""
        item['prediction'] = prediction
        item['thinking'] = thinking
        item['context'] = item['context'][:512]
        return item

    def process_data_subset(self, data_subset: List[Dict[str, Any]], output_file: str, file_lock: mp.Lock) -> None:
        """process data subset"""
        for item in tqdm(data_subset, desc=f"process data subset"):
            result = self.process_single_item(item)
            # Use process lock to ensure atomic file writing
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fout.flush()
    
    def infer(self, data_subsets: List[List[Dict[str, Any]]], output_file: str) -> None:
        """infer"""
        logger.info(f"Starting {len(data_subsets)} processes for parallel processing...")
        processes: List[mp.Process] = []  # Fixed undefined processes variable
        file_lock = mp.Lock()  # Create process lock
        
        for data_subset in data_subsets:
            p = mp.Process(
                target=self.process_data_subset, 
                args=(data_subset, output_file, file_lock)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        logger.info("All processes completed successfully")
