import numpy as np
import pandas as pd
import faiss
import csv
import datasets

from rich.progress import Progress, track
from rich import print as rprint
from rich.pretty import pprint



if __name__ == "__main__":

    records = list()
    with open("./data/wiki_simple.csv", "r") as fp:
        reader = csv.DictReader(fp)

        for row in track(reader):
            records.append(row)

    hf_data = datasets.Dataset.from_list(records)
    hf_data.save_to_disk("./data/hf_wiki_data")
