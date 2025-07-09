import os
import json
import requests
import pandas as pd
from typing import List, Dict
from datasets import load_dataset

TURNS2K_URL = "https://huggingface.co/datasets/Alireza1044/TURNS-2K/resolve/main/TURNS-2K.json"
DATA_DIR = "./data"
RAW_JSON = os.path.join(DATA_DIR, "TURNS-2K.json")
BERT_CSV = os.path.join(DATA_DIR, "turns2k_bert.csv")
OPENAI_JSONL = os.path.join(DATA_DIR, "turns2k_openai.jsonl")

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def load_turns2k_dataset():
    """Load TURNS-2K dataset using HuggingFace datasets library."""
    try:
        dataset = load_dataset("latishab/turns-2k")
        print(f"✓ Successfully loaded {len(dataset)} conversations from HuggingFace")
        return dataset
    except Exception as e:
        print(f"✗ Error loading TURNS-2K dataset: {e}")
        print("Make sure you have a HuggingFace account and are authenticated if required.")
        raise

def parse_turns2k_for_bert(dataset) -> pd.DataFrame:
    """Parse TURNS-2K HuggingFace dataset for BERT training: columns 'text', 'label' (1=END, 0=NOT_END)."""
    df = dataset.to_pandas()
    print(f"✓ Parsed {len(df)} turns for BERT training from HuggingFace dataset")
    return df


def parse_turns2k_for_openai(dataset) -> List[Dict]:
    """Parse TURNS-2K HuggingFace dataset for OpenAI fine-tuning: JSONL with 'messages' and 'end' label."""
    jsonl_rows = []
    for row in dataset:
        messages = [{"role": "user", "content": row['text'].strip()}]
        jsonl_rows.append({"messages": messages, "end": bool(row['label'])})
    print(f"✓ Parsed {len(jsonl_rows)} conversations for OpenAI fine-tuning from HuggingFace dataset")
    return jsonl_rows


def export_bert_csv(df: pd.DataFrame, out_path: str = BERT_CSV):
    df.to_csv(out_path, index=False)
    print(f"✓ Exported BERT training CSV to {out_path}")

def export_openai_jsonl(jsonl_rows: List[Dict], out_path: str = OPENAI_JSONL):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✓ Exported OpenAI fine-tuning JSONL to {out_path}")

def load_ubuntu_dialogue_corpus(split: str = "train"):
    """Load the Ubuntu Dialogue Corpus using HuggingFace datasets library."""
    try:
        dataset = load_dataset("ubuntu_dialogs_corpus", split=split, trust_remote_code=True)
        print(f"✓ Successfully loaded Ubuntu Dialogue Corpus split '{split}' with {len(dataset)} dialogues from HuggingFace")
        return dataset
    except Exception as e:
        print(f"✗ Error loading Ubuntu Dialogue Corpus: {e}")
        print("Make sure you have a HuggingFace account and are authenticated if required.")
        raise

def main():
    print("=== TURNS-2K Preparation Script ===")
    # dataset = load_turns2k_dataset()
    dataset = load_ubuntu_dialogue_corpus()
    if isinstance(dataset, dict):  # DatasetDict
        dataset = dataset['train']
    df_bert = parse_turns2k_for_bert(dataset)
    export_bert_csv(df_bert)
    jsonl_rows = parse_turns2k_for_openai(dataset)
    export_openai_jsonl(jsonl_rows)
    print("All done!")

if __name__ == "__main__":
    main() 