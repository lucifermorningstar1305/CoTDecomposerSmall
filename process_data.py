import pandas as pd
import os
import json
import re

from rich.progress import track
from datasets import Dataset, DatasetDict
if __name__ == "__main__":
    
    with open("./data/hotpot_qa.json", "r") as fp:
        json_data = json.load(fp)


    print(f"Initial Data Length: {len(json_data['question'])}")
    data = []

    for ques, decomp, pred_ans, ans in track(zip(json_data["question"], json_data["decomposition"], json_data["pred_ans"], json_data["answer"])):

        pred_ans = re.findall(r"(\(\w\)\s\w+)", pred_ans)
        if len(pred_ans):
            pred_ans = pred_ans[0]
            pred_ans = re.sub(r"(\(\w\))", "", pred_ans).strip()
            
            if pred_ans == ans:
                data.append({
                    "question": ques,
                    "decomposition": decomp,
                    "predicted_answer": pred_ans,
                    "answer": ans
                })

    print(f"Final Data after filtering: {len(data)}")
    print(f"A sample of the data:\n{data[0]}")

    data = Dataset.from_pandas(pd.DataFrame(data=data))
    data.push_to_hub("bvk1ng/hotpotqa_cot_decomposed")
    

    # sample_decomp = data[0]["decomposition"]
    # print(re.findall(r"(<sub_q>.*</sub_q>)", sample_decomp))