import torch
import lightning.pytorch as pl
import os
import argparse

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from transformers import BertTokenizer, BartTokenizerFast
from datasets import load_dataset
from data_processing import LitCotDataModule
from modelling import LitRAG

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--faiss", "-f", required=False, type=str, default="./indices/wiki_index.index", help="location of faiss indices")
    parser.add_argument("--wiki", "-w", required=False, type=str, default="./data/hf_wiki_data", help="location of wiki dataset for retrieval")
    parser.add_argument("--val_size", "-v", required=False, type=float, default=.1, help="percentage of data to be used for validation")
    parser.add_argument("--batch_size", "-B", required=False, type=int, default=8, help="training batch size")
    parser.add_argument("--val_batch_size", "-b", required=False, type=int, default=16, help="validation batch size")
    parser.add_argument("--patience", "-p", required=False, type=int, default=10, help="early stopping patience level (in epochs)")
    parser.add_argument("--chkpt_name", "-c", required=False, type=str, default="rag_chkpt", help="saving name for model checkpoints")
    parser.add_argument("--run_name", "-r", required=False, type=str, default="rag", help="saving name of model runs for W&B")
    parser.add_argument("--num_workers", "-n", required=False, type=int, default=2, help="number of workers for spawning for torch dataloader")

    args = parser.parse_args()
    

    if not os.path.exists("./chkpts"):
        os.mkdir("./chkpts")

    n_gpus = torch.cuda.device_count()

    cot_dataset = load_dataset("bvk1ng/hotpotqa_cot_decomposed", split="train", cache_dir="./data/")
    cot_dataset = cot_dataset.shuffle(seed=42).train_test_split(test_size=args.val_size, shuffle=False)

    ques_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    generator_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    train_dm = LitCotDataModule(cot_dataset["train"], 
                                ques_tokenizer=ques_tokenizer, 
                                generator_tokenizer=generator_tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)
    
    val_dm = LitCotDataModule(cot_dataset["test"], 
                                ques_tokenizer=ques_tokenizer, 
                                generator_tokenizer=generator_tokenizer, batch_size=args.val_batch_size, num_workers=args.num_workers)
    
    train_dm.setup(stage="train")
    val_dm.setup(stage="val")

    train_dl = train_dm.train_dataloader()
    val_dl = val_dm.val_dataloader()

    model = LitRAG(args.faiss, args.wiki)

    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True)
    prg_bar = RichProgressBar()
    chkptng = ModelCheckpoint(monitor="val_loss",
                              mode="min",
                              save_on_train_epoch_end=False,
                              verbose=True,
                              filename=args.chkpt_name,
                              dirpath="chkpts")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        precision=16,
        logger = WandbLogger(name=args.run_name, project="teaching_small_models"),
        max_epochs=100,
        min_epochs=5,
        strategy="ddp" if n_gpus > 1 else "auto",
        callbacks=[early_stop, chkptng, prg_bar]
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)