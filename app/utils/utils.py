import logging
import os
import platform
import psutil
import pyodbc
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from typing import Dict, Any

# Global flag to track if logging has been initialized
_logging_initialized = False

def get_db_connection(db_config: Dict[str, str]):
    """Create and return a database connection"""
    try:
        conn_str = ';'.join([f"{k}={v}" for k, v in db_config.items()])
        conn = pyodbc.connect(conn_str, timeout=30)  # 30 second timeout
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        raise

def predict_convo_ending(text, model, tokenizer, device='cpu'):
    """Predict if conversation should end using the BERT model"""
    import torch
    
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