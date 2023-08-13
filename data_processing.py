from typing import Dict, List, Tuple, Any, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import torch
import torch.utils.data as td
import lightning.pytorch as pl
import faiss
import os


class CotDataModule(td.Dataset):
    def __init__(self, cot_dataset:Dict, ques_tokenizer: Any, generator_tokenizer: Any):

        self.cot_dataset = cot_dataset
        self.ques_tokenizer = ques_tokenizer
        self.generator_tokenizer = generator_tokenizer

    def __len__(self) -> int:
        return len(self.cot_dataset)
    
    def __getitem__(self, index) -> Any:
        
        record = self.cot_dataset[index]
        question = record["question"]
        decomposition = record["decomposition"]
        answer = record["answer"]

        inputs = self.ques_tokenizer(question, 
                                     max_length=256, truncation=True, padding="max_length",
                                     return_tensors="pt")
        
        outputs = self.generator_tokenizer(decomposition+f"\n [Ans] {answer}", 
                                           max_length=1024, truncation=True, padding="max_length",
                                           return_tensors="pt")
        

        return {
            "input_ids": inputs["input_ids"],
            "input_attention_mask": inputs["attention_mask"],
            "output_ids": outputs["input_ids"],
            "output_attention_mask": outputs["attention_mask"]
        }
    
class LitCotDataModule(pl.LightningDataModule):
    def __init__(self, cot_dataset: Dict, ques_tokenizer: Any, generator_tokenizer: Any,
                 batch_size: Optional[int]=32, num_workers: Optional[int]=4):
        
        self.cot_dataset = cot_dataset
        self.ques_tokenizer = ques_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None

    def setup(self, stage: str) -> None:

        if stage not in ["train", "val", "test"]:
            raise Exception("Expected stage to be either train/val/test")
        
        if stage == "train":
            self.dataset = CotDataModule(self.cot_dataset, self.ques_tokenizer,
                                         self.generator_tokenizer)
            
        elif stage == "val":
            self.dataset = CotDataModule(self.cot_dataset, self.ques_tokenizer, 
                                         self.generator_tokenizer)
            
        else:
            self.dataset = CotDataModule(self.cot_dataset, self.ques_tokenizer, 
                                         self.generator_tokenizer)
            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return td.DataLoader(self.dataset, batch_size=self.batch_size, 
                             shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return td.DataLoader(self.dataset, batch_size=self.batch_size, 
                             shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return td.DataLoader(self.dataset, batch_size=self.batch_size, 
                             shuffle=True, num_workers=self.num_workers)
    


        
    


        


    

