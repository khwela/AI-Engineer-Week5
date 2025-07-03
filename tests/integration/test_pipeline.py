import unittest
import torch
import tempfile
import json
import os
from transformers import AutoTokenizer

from src.models.academic_summarizer import create_summarizer
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls._create_test_data()
        
        # Initialize model and tokenizer
        cls.model = create_summarizer()
        cls.model.to(cls.device)
        cls.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_data(cls):
        """Create test dataset files."""
        test_data = [
            {
                "text": "This is a test paper about machine learning. "
                        "It discusses important concepts and methodologies. "
                        "The results show significant improvements.",
                "summary": "A paper discussing machine learning concepts with positive results.",
                "citations": [
                    {"text": "Author et al., 2023", "context": "Previous work"}
                ]
            }
            for _ in range(5)  # Create 5 examples
        ]
        
        # Save test data for each split
        for split in ["train", "val", "test"]:
            with open(os.path.join(cls.temp_dir, f"{split}.json"), "w") as f:
                json.dump(test_data, f)
    
    def test_complete_pipeline(self):
        """Test the complete training and evaluation pipeline."""
        # Create dataloaders
        dataloaders = create_dataloaders(
            data_dir=self.temp_dir,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=512,
            summary_max_length=128,
            num_workers=0  # Use 0 for testing
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            config={
                "num_epochs": 1,  # Use 1 epoch for testing
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "output_dir": os.path.join(self.temp_dir, "checkpoints"),
                "use_wandb": False
            }
        )
        
        # Train for one epoch
        trainer.train()
        
        # Create evaluator
        evaluator = Evaluator(tokenizer=self.tokenizer)
        
        # Evaluate on test set
        metrics = evaluator.evaluate(self.model, dataloaders["test"])
        
        # Check metrics
        self.assertIn("rouge_1", metrics)
        self.assertIn("rouge_2", metrics)
        self.assertIn("rouge_l", metrics)
        self.assertIn("bleu", metrics)
        self.assertIn("bertscore_f1", metrics)
    
    def test_data_loading(self):
        """Test data loading and batch formation."""
        dataloaders = create_dataloaders(
            data_dir=self.temp_dir,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=512,
            summary_max_length=128,
            num_workers=0
        )
        
        # Check if dataloaders are created for all splits
        self.assertIn("train", dataloaders)
        self.assertIn("val", dataloaders)
        self.assertIn("test", dataloaders)
        
        # Check batch format
        batch = next(iter(dataloaders["train"]))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("target_ids", batch)
        
        # Check batch shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)  # batch_size
        self.assertEqual(batch["attention_mask"].shape[0], 2)
        self.assertEqual(batch["target_ids"].shape[0], 2)
    
    def test_model_saving_loading(self):
        """Test model checkpoint saving and loading."""
        # Train model and save checkpoint
        checkpoint_dir = os.path.join(self.temp_dir, "test_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        dataloaders = create_dataloaders(
            data_dir=self.temp_dir,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=512,
            summary_max_length=128,
            num_workers=0
        )
        
        trainer = Trainer(
            model=self.model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            config={"output_dir": checkpoint_dir, "use_wandb": False}
        )
        
        # Save checkpoint
        trainer._save_checkpoint(epoch=0, global_step=100)
        
        # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-100.pt")
        loaded_model = Trainer.load_checkpoint(self.model, checkpoint_path)
        
        # Check if model parameters are identical
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
    
    def test_inference(self):
        """Test model inference."""
        test_text = """
        This is a comprehensive study on deep learning applications.
        We propose a novel architecture that improves performance.
        Our experiments show significant improvements over baselines.
        """
        
        # Generate summary
        summary = self.model.summarize(test_text)
        
        # Check summary
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        
        # Test with citations
        citations = [
            {"text": "Author et al., 2023", "context": "Related work"}
        ]
        summary_with_citations = self.model.summarize(test_text, citations)
        
        self.assertIsInstance(summary_with_citations, str)
        self.assertTrue(len(summary_with_citations) > 0)

if __name__ == "__main__":
    unittest.main() 