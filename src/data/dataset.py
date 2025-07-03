import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import json
import os
from transformers import PreTrainedTokenizer
import numpy as np

class AcademicPaperDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        summary_max_length: int = 256,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            tokenizer: Tokenizer for processing text
            max_length: Maximum length of input text
            summary_max_length: Maximum length of summary
            split: Dataset split ('train', 'val', or 'test')
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        
        # Load data
        self.data = self._load_data(split)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        item = self.data[idx]
        
        # Tokenize input text
        inputs = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target summary if available
        if "summary" in item:
            targets = self.tokenizer(
                item["summary"],
                max_length=self.summary_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Process citations if available
            if "citations" in item:
                citations = self._process_citations(item["citations"])
            else:
                citations = None
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "target_ids": targets["input_ids"].squeeze(0),
                "citations": citations
            }
        
        # For inference, return only inputs
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
    
    def _load_data(self, split: str) -> List[Dict]:
        """Load data from files."""
        data_file = os.path.join(self.data_dir, f"{split}.json")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def _process_citations(self, citations: List[Dict]) -> torch.Tensor:
        """Process citations into tensor format."""
        # Placeholder for citation processing
        # This would convert citation dictionaries into tensor features
        return torch.zeros((self.max_length, 768))  # Dummy implementation

class DataCollator:
    def __init__(self, pad_token_id: int):
        """
        Initialize the data collator.
        
        Args:
            pad_token_id: ID of the padding token
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            features: List of dictionaries containing tensors
        
        Returns:
            Batch dictionary with padded tensors
        """
        # Get batch size and max lengths
        batch_size = len(features)
        max_input_len = max(f["input_ids"].size(0) for f in features)
        
        # Initialize padded tensors
        input_ids = torch.full(
            (batch_size, max_input_len),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_input_len),
            dtype=torch.long
        )
        
        # Fill padded tensors
        for i, feature in enumerate(features):
            input_len = feature["input_ids"].size(0)
            input_ids[i, :input_len] = feature["input_ids"]
            attention_mask[i, :input_len] = feature["attention_mask"]
        
        # Process target ids if available
        if "target_ids" in features[0]:
            max_target_len = max(f["target_ids"].size(0) for f in features)
            target_ids = torch.full(
                (batch_size, max_target_len),
                self.pad_token_id,
                dtype=torch.long
            )
            
            for i, feature in enumerate(features):
                target_len = feature["target_ids"].size(0)
                target_ids[i, :target_len] = feature["target_ids"]
        else:
            target_ids = None
        
        # Process citations if available
        if "citations" in features[0] and features[0]["citations"] is not None:
            citations = torch.stack([f["citations"] for f in features])
        else:
            citations = None
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "citations": citations
        }

def create_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    summary_max_length: int = 256,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset files
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for DataLoaders
        max_length: Maximum length of input text
        summary_max_length: Maximum length of summary
        num_workers: Number of worker processes for data loading
    
    Returns:
        Dictionary containing DataLoaders for each split
    """
    # Create datasets
    datasets = {
        split: AcademicPaperDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            summary_max_length=summary_max_length,
            split=split
        )
        for split in ["train", "val", "test"]
    }
    
    # Create data collator
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        for split, dataset in datasets.items()
    }
    
    return dataloaders 