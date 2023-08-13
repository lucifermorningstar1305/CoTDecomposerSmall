from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import faiss
import codecs
import csv
import os

from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, BertTokenizer
from sentence_transformers import SentenceTransformer
from rich.progress import track

def process_chunks(facts: str, chunk_size: Optional[int]=100):
    """ Function to process the facts into 100 words chunks """
    
    tokenized_facts = facts.split()
    chunks = [tokenized_facts[i:i+chunk_size] for i in range(0, len(tokenized_facts), chunk_size)]
    chunked_txt = [" ".join(chunk) for chunk in chunks]
    
    return chunked_txt
    

def get_embeddings(chunks: List):
    """ Function to return the embeddings of the chunk. """

    # inputs = tokenizer(chunks, max_length=128, truncation=True,
    #                    padding="max_length", return_tensors="pt").to(device)
    
    # with torch.no_grad():
    #     cls_embedding = model(**inputs).pooler_output
    #     cls_embedding = cls_embedding.detach().cpu().numpy()
    embedding = model.encode(chunks)
    
    return embedding
    

if __name__ == "__main__":

    wiki_dataset = load_dataset("wikipedia", "20220301.simple", split="train", cache_dir="./data/")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device).eval()
    model = SentenceTransformer("sentence-transformers/facebook-dpr-ctx_encoder-multiset-base").to(device).eval()

    index = faiss.IndexHNSWFlat(768, 32, faiss.METRIC_INNER_PRODUCT)
    
    with codecs.open("./data/wiki_simple.csv", "w", "utf-8") as fp:
        
        csv_writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["title", "chunk"])

        for i in track(range(len(wiki_dataset))):
            
            elem = wiki_dataset[i]

            title = elem["title"]
            chunks = process_chunks(elem["text"])
            new_chunks = list(map(lambda x: title + ": "+x, chunks))
            
            embedding = get_embeddings(new_chunks)
            index.add(embedding)
            [csv_writer.writerow([title, chunk]) for chunk in chunks]

            chunks.clear()
            new_chunks.clear()
    
    if not os.path.exists("./indices"):
        os.mkdir("./indices")

    faiss.write_index(index, "./indices/wiki_index.index")