import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
import platform
import psutil
import platform
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

# Example data: List of conversation snippets and labels
# data = {
#     "text": [
#         "User: Thanks for your help. Agent: You're welcome, have a great day.",
#         "User: I still need help with my issue. Agent: Sure, what else?",
#     ],
#     "label": [1, 0],  # 1 = End, 0 = Not End
# }
# dataset = Dataset.from_dict(data)

def setup_device_and_logging():
    """Setup device with Apple Silicon optimization"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"System: {platform.platform()}")
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🚀 Using Apple Silicon MPS (Metal Performance Shaders)")
        logger.info("Expected 3-5x speedup over CPU!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using NVIDIA CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("⚠️  Using CPU only - consider upgrading PyTorch for MPS support")
    
    # Memory info
    if hasattr(psutil, 'virtual_memory'):
        memory = psutil.virtual_memory()
        logger.info(f"Available RAM: {memory.total // (1024**3)}GB")
    
    return device

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def predict_convo_ending(text, model, tokenizer, device='cpu'):
    """Predict if conversation should end"""
    model.eval()
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    return {
        'prediction': "END" if prediction == 1 else "NOT_END",
        'confidence': confidence,
        'probabilities': {
            'NOT_END': probs[0][0].item(),
            'END': probs[0][1].item()
        }
    }

# # Load DailyDialog as example
# dataset = load_dataset("daily_dialog", split="train")
def gather_chat_endings(daily_dialog_dataset):
    chat_endings = []
    for convo in daily_dialog_dataset['dialog']:
        chat_endings.append(convo[-1])
    return chat_endings

        # num_turns = len(convo)
        # for i, turn in enumerate(convo):
        #     label = 1 if i == num_turns - 1 else 0  # Label last turn as END
        #     chat_endings.append({
        #         "text": " ".join(convo[:i+1]), 

def preprocess_conversations_for_ending_cues(daily_dialog_dataset, end_window_size=1, non_end_window_size=3):
    """
    Improved preprocessing to better capture explicit conversation ending cues.
    Focuses on the very last turn for 'END' and earlier, more ambiguous turns for 'NOT_END'.
    
    Args:
        daily_dialog_dataset (Dataset): The DailyDialog dataset.
        end_window_size (int): Number of turns from the end of the conversation to consider as 'END'.
                                A smaller size (e.g., 1 or 2) is often better for explicit endings.
        non_end_window_size (int): Number of turns for 'NOT_END' examples, taken from earlier parts.
    """
    processed_data = []

    for convo_obj in daily_dialog_dataset['dialog']:
        convo_turns = convo_obj # Assuming convo_obj is already a list of strings (turns)
        num_turns = len(convo_turns)

        # 1. Generate Positive Examples (END)
        # Prioritize the very last turn or last few turns
        if num_turns >= end_window_size:
            # Use only the last `end_window_size` turns for explicit endings
            final_segment = " ".join(convo_turns[-end_window_size:])
            processed_data.append({
                "text": final_segment,
                "label": 1  # END
            })
            
            # Optionally, for very short conversations, ensure at least one 'END' example
            if num_turns == 1 and end_window_size > 1:
                # If convo is just one turn, and end_window_size > 1, adjust
                processed_data.append({
                    "text": convo_turns[0],
                    "label": 1
                })


        # 2. Generate Negative Examples (NOT_END)
        # These should clearly NOT be endings.
        # Strategy A: Random segments from the beginning/middle (excluding the last part)
        # Ensure we have enough turns to form a non-ending segment without overlapping the end.
        if num_turns > non_end_window_size:
            # Choose a random start index that leaves enough room for a non_end_window_size segment
            # and does not include the very end of the conversation.
            max_non_end_start_index = num_turns - non_end_window_size - 1 # -1 to avoid last turn entirely
            if max_non_end_start_index >= 0:
                start_index = np.random.randint(0, max_non_end_start_index + 1)
                non_end_segment = " ".join(convo_turns[start_index : start_index + non_end_window_size])
                processed_data.append({
                    "text": non_end_segment,
                    "label": 0  # NOT_END
                })
        
        # Strategy B: Explicit "non-ending" patterns (e.g., questions, offers of help)
        # This requires domain knowledge or analyzing dialogue acts, but can be powerful.
        # For DailyDialog, we might look for common starting phrases or questions.
        if num_turns >= non_end_window_size and len(convo_turns[0].strip()) > 0:
            # Example: Taking the very first turn(s) as typically non-ending
            # Only if the conversation is long enough to have distinct start/end
            if num_turns > end_window_size + non_end_window_size: # Ensures distinct start and end
                initial_segment = " ".join(convo_turns[:non_end_window_size])
                processed_data.append({
                    "text": initial_segment,
                    "label": 0  # NOT_END
                })
                
    return pd.DataFrame(processed_data)

def preprocess_conversations_v2(dataset, window_size=3):
    """Preprocess conversations for training"""
    processed_data = []
    
    for convo in dataset['dialog']:
        if len(convo) >= window_size:
            # Take last N turns as positive example (END)
            last_turns = " ".join(convo[-window_size:])
            processed_data.append({
                "text": last_turns,
                "label": 1  # END
            })
            
            # Take middle N turns as negative example (NOT END)
            if len(convo) > window_size * 2:
                mid_start = len(convo) // 2 - window_size // 2
                mid_turns = " ".join(convo[mid_start:mid_start + window_size])
                processed_data.append({
                    "text": mid_turns,
                    "label": 0  # NOT END
                })
    
    return pd.DataFrame(processed_data)

def load_daily():
    """Load DailyDialog dataset"""
    try:
        logger.info("Loading DailyDialog dataset...")
        dataset = load_dataset("daily_dialog", split="train")
        logger.info(f"Loaded {len(dataset)} conversations")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def load_custom_jsonl(file_path):
    """Load custom JSONL dataset with conversation ending data"""
    import json
    
    try:
        logger.info(f"Loading custom JSONL dataset from {file_path}...")
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_obj = json.loads(line.strip())
                    if 'messages' in json_obj:
                        # Extract user message and assistant response
                        user_message = None
                        assistant_response = None
                        
                        for msg in json_obj['messages']:
                            if msg['role'] == 'user':
                                user_message = msg['content']
                            elif msg['role'] == 'assistant':
                                assistant_response = msg['content']
                        
                        if user_message and assistant_response:
                            # Convert assistant response to binary label
                            label = 1 if assistant_response.strip() == '[END]' else 0
                            data.append({
                                'text': user_message,
                                'label': label
                            })
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} examples from custom JSONL file")
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error loading custom JSONL dataset: {e}")
        raise

def tokenize_function(examples, tokenizer):
    """Tokenize examples using the provided tokenizer while preserving labels"""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128,
        return_tensors=None
    )
    
    # Preserve the labels - HuggingFace expects "labels" not "label"
    tokenized["labels"] = examples["label"]
    
    return tokenized

def setup_device_and_logging():
    """Setup device with Apple Silicon optimization"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"System: {platform.platform()}")
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🚀 Using Apple Silicon MPS (Metal Performance Shaders)")
        logger.info("Expected 3-5x speedup over CPU!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using NVIDIA CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("⚠️  Using CPU only - consider upgrading PyTorch for MPS support")
    
    # Memory info
    if hasattr(psutil, 'virtual_memory'):
        memory = psutil.virtual_memory()
        logger.info(f"Available RAM: {memory.total // (1024**3)}GB")
    
    return device

def main():
    # Apple Silicon M4 optimized:
    device = setup_device_and_logging()
    
    # Load dataset
    # OLD: DailyDialog dataset (commented out)
    # daily_dialog_dataset = load_daily()
    # processed_df = preprocess_conversations_v2(daily_dialog_dataset)
    # processed_df = preprocess_conversations_for_ending_cues(daily_dialog_dataset, end_window_size=1, non_end_window_size=3) # NEW
    
    # NEW: Load custom JSONL dataset
    processed_df = load_custom_jsonl("conversation_ending_data_shuffled.jsonl")

    
    logger.info("Dataset preprocessing complete")
    logger.info(f"Dataset shape: {processed_df.shape}")
    logger.info(f"Label distribution:\n{processed_df['label'].value_counts()}")
    
    # Convert pandas DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(processed_df)
    
    # Debug: Check dataset columns before tokenization
    logger.info(f"Dataset columns before tokenization: {dataset.column_names}")
    logger.info(f"Sample data: {dataset[0]}")
    
    # Initialize tokenizer and model with Context7 approach
    model_name = "bert-base-uncased"
    logger.info(f"Loading tokenizer and model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Move model to device
    model.to(device)
    
    # Apply tokenization - Context7 style with proper label handling
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True,
        remove_columns=["text"]  # Only remove text column, keep labels
    )
    
    # Debug: Check dataset columns after tokenization
    logger.info(f"Dataset columns after tokenization: {tokenized_dataset.column_names}")
    logger.info(f"Sample tokenized data: {tokenized_dataset[0]}")
    
    # Verify labels are present
    if "labels" not in tokenized_dataset.column_names:
        raise ValueError("Labels not found in tokenized dataset. Check tokenization process.")
    
    # Split into train/eval
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Debug: Check final dataset structure
    logger.info(f"Train dataset columns: {train_dataset.column_names}")
    sample_batch = train_dataset[:2]
    logger.info(f"Sample batch keys: {sample_batch.keys()}")
    
    # Configure training arguments - Context7 style
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        warmup_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=None,
        seed=42,
        fp16=False,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Starting training...")
    print("\n" + "="*50)
    print("STARTING TRAINING...")
    print("="*50)
    
    # Train the model
    trainer.train()

    print("\n" + "="*50)
    print("TRAINING COMPLETE! EVALUATING...")
    print("="*50)
    
    # Get final evaluation metrics
    eval_results = trainer.evaluate()
    logger.info("Final evaluation complete")
    
    print("\nFinal Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save the model
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    logger.info("Model saved to ./final_model")
    
    # Test predictions
    test_examples = [
        "Thanks for your help. Have a good day!",
        "How can I help you today?",
        "Goodbye, see you later!",
        "What's the weather like?",
        "Thank you so much. Bye!"
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS...")
    print("="*50)
    
    for text in test_examples:
        result = predict_convo_ending(text, model, tokenizer, device)
        print(f"\nText: '{text}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: NOT_END={result['probabilities']['NOT_END']:.4f}, END={result['probabilities']['END']:.4f}")

    # Diagnostic information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU will be used for training!")
        logger.info(f"MPS device: {device}")
    else:
        logger.info("MPS not available, check PyTorch version")

if __name__ == "__main__":
    
    main()
