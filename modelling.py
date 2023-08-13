from typing import Any, List, Dict, Tuple, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import faiss
import lightning.pytorch as pl
import re

from transformers import BertModel, BartForConditionalGeneration, BartTokenizerFast, BertTokenizer
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType


class LitRAG(pl.LightningModule):
    def __init__(self, faiss_index_path: str, retrieval_dataset_path: str, max_length: Optional[int]=300, n_docs: Optional[int]=5):

        super().__init__()

        ques_model = BertModel.from_pretrained("bert-base-uncased")
        ques_peft_config = LoraConfig(inference_mode=False, r=4, lora_alpha=16, lora_dropout=.25)
        self.ques_model = get_peft_model(ques_model, ques_peft_config)
        self.ques_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        generator_peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=4, lora_alpha=16, lora_dropout=.25)
        self.generator_model = get_peft_model(generator_model, generator_peft_config)
        self.generator_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        self.max_length = max_length
        self.rouge_score = evaluate.load("rouge")
        self.n_docs = n_docs

        self.doc_index = faiss.read_index(faiss_index_path)
        self.doc_index.hnsw.efSearch = 400
        self.retrieval_dataset = load_from_disk(retrieval_dataset_path)

    def post_process_docs(self, title: str, chunk: str, question: str) -> str:

        out = title + " // " + chunk + " / " + question
        return out.replace("  ", " ") 


    def forward(self, input_ids: torch.Tensor, input_attention_mask: torch.Tensor, 
                output_ids: torch.Tensor, output_attention_mask: torch.Tensor) -> torch.Tensor:
            
        input_ids, input_attention_mask = input_ids.squeeze(), input_attention_mask.squeeze()
        
        ques_embedding = self.ques_model(input_ids=input_ids, attention_mask=input_attention_mask).pooler_output
        _, indices_mat = self.doc_index.search(x=ques_embedding.detach().cpu().numpy(), k=self.n_docs)

        questions = self.ques_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        ctx_docs = [self.post_process_docs(self.retrieval_dataset[i]["title"], 
                                      self.retrieval_dataset[i]["chunk"],
                                      questions[idx]) for idx, indices in enumerate(indices_mat) for i in indices.tolist()]
        
        ctx_inputs = self.generator_tokenizer(ctx_docs, max_length=self.max_length,
                                                               padding="max_length", truncation=True, return_tensors="pt")
        
        output_ids = output_ids.repeat_interleave(self.n_docs, dim=0).squeeze()
        # print(ctx_inputs["attention_mask"].shape, output_attention_mask.shape, output_attention_mask.repeat_interleave(5, dim=0).shape)
        output_attention_mask = output_attention_mask.repeat_interleave(self.n_docs, dim=0).squeeze()

        logits = self.generator_model(
            input_ids = ctx_inputs["input_ids"].to(output_ids), 
            attention_mask = ctx_inputs["attention_mask"].to(output_attention_mask),
            decoder_input_ids = output_ids,
            decoder_attention_mask = output_attention_mask
        ).logits

        return logits
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        logits = torch.argmax(logits, dim=-1)
        return F.cross_entropy(logits, labels)
    
    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple:

        logits = torch.argmax(logits, dim=-1)

        predictions = self.generator_tokenizer.batch_decode(logits, skip_special_tokens=True, 
                                                               clean_up_tokenization_spaces=True)
        
        references = self.generator_tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)
        
        rouge_scores = self.rouge_score.compute(predictions=predictions, references=references)

        prediction_ans = [re.findall(r"(\[Ans\]\s(yes|no))", pred)[0][-1] for pred in predictions]
        reference_ans = [re.findall(r"(\[Ans\]\s(yes|no))", ref)[0][-1] for ref in references]

        acc = sum([i == j for i, j in zip(prediction_ans, reference_ans)]) / len(reference_ans)

        return rouge_scores, acc
    

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:
        
        query_input_ids, query_attention_mask = batch["input_ids"], batch["input_attention_mask"]
        cot_input_ids, cot_attention_mask = batch["output_ids"], batch["output_attention_mask"]


        logits = self(query_input_ids, query_attention_mask, 
                      cot_input_ids, cot_attention_mask)
        
        loss = self._compute_loss(logits, cot_input_ids)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        return {
            "loss": loss
        }
    
    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:

        query_input_ids, query_attention_mask = batch["input_ids"], batch["input_attention_mask"]
        cot_input_ids, cot_attention_mask = batch["output_ids"], batch["output_attention_mask"]


        logits = self(query_input_ids, query_attention_mask, 
                      cot_input_ids, cot_attention_mask)
        
        loss = self._compute_loss(logits, cot_input_ids)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True)

        rouge_scores, acc= self._compute_metrics(logits, cot_input_ids)

        self.log("val_acc", acc, prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True)

        for key, val in rouge_scores.items():
            self.log("val_"+key, val, prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, eta_min=1e-6)

        return {
            "optimizer": optimizer, 
            "scheduler": scheduler
        }

    

        

        
                
