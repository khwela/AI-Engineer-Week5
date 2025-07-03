import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from typing import Dict, Any, Optional

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set default config if none provided
        self.config = {
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "num_epochs": 10,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "logging_steps": 100,
            "save_steps": 1000,
            "output_dir": "checkpoints",
            "use_wandb": True
        }
        if config:
            self.config.update(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Initialize wandb if enabled
        if self.config["use_wandb"]:
            wandb.init(project="academic-summarizer", config=self.config)
    
    def train(self):
        """Train the model."""
        best_val_loss = float("inf")
        global_step = 0
        
        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_dataloader, desc="Training")
            for batch in progress_bar:
                loss = self._training_step(batch)
                
                # Gradient accumulation
                loss = loss / self.config["gradient_accumulation_steps"]
                loss.backward()
                
                train_loss += loss.item()
                train_steps += 1
                
                # Update weights
                if train_steps % self.config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # Logging
                if global_step % self.config["logging_steps"] == 0:
                    avg_loss = train_loss / train_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
                    if self.config["use_wandb"]:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "global_step": global_step
                        })
                
                # Save checkpoint
                if global_step % self.config["save_steps"] == 0:
                    self._save_checkpoint(epoch, global_step)
            
            # Validation
            if self.val_dataloader:
                val_loss = self._validate()
                print(f"\nValidation loss: {val_loss:.4f}")
                
                if self.config["use_wandb"]:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch + 1
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, global_step, is_best=True)
    
    def _training_step(self, batch) -> torch.Tensor:
        """Perform one training step."""
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            citations=batch.get("citations"),
            target_ids=batch["target_ids"].to(self.model.device)
        )
        return outputs["loss"]
    
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.model.device),
                    attention_mask=batch["attention_mask"].to(self.model.device),
                    citations=batch.get("citations"),
                    target_ids=batch["target_ids"].to(self.model.device)
                )
                val_loss += outputs["loss"].item()
                val_steps += 1
        
        return val_loss / val_steps
    
    def _save_checkpoint(self, epoch: int, global_step: int, is_best: bool = False):
        """Save a model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        
        if is_best:
            path = os.path.join(self.config["output_dir"], "best_model.pt")
        else:
            path = os.path.join(self.config["output_dir"], f"checkpoint-{global_step}.pt")
        
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint: {path}")
    
    @classmethod
    def load_checkpoint(cls, model: nn.Module, checkpoint_path: str):
        """Load a model from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model 