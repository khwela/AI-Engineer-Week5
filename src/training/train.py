import argparse
import os
import torch
from transformers import AutoTokenizer

from src.models.academic_summarizer import create_summarizer
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Train the academic paper summarizer")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Encoder model name or path"
    )
    parser.add_argument(
        "--decoder_model",
        type=str,
        default="facebook/bart-large",
        help="Decoder model name or path"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--summary_max_length",
        type=int,
        default=256,
        help="Maximum summary sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases logging"
    )
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    
    # Create model
    model = create_summarizer({
        "encoder_model": args.encoder_model,
        "decoder_model": args.decoder_model,
        "max_length": args.max_length
    })
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        summary_max_length=args.summary_max_length
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        config={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "warmup_steps": args.warmup_steps,
            "output_dir": args.output_dir,
            "use_wandb": args.use_wandb
        }
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Create evaluator
    evaluator = Evaluator(tokenizer=tokenizer)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = evaluator.evaluate(model, dataloaders["test"])
    
    # Print results
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 