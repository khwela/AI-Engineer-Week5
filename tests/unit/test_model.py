import unittest
import torch
from src.models.academic_summarizer import (
    AcademicSummarizer,
    CitationProcessor,
    SectionClassifier,
    create_summarizer
)

class TestAcademicSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = create_summarizer()
        cls.model.to(cls.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, AcademicSummarizer)
        self.assertIsInstance(self.model.citation_handler, CitationProcessor)
        self.assertIsInstance(self.model.section_classifier, SectionClassifier)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        # Create dummy input
        batch_size = 2
        seq_length = 512
        input_ids = torch.randint(
            0, 1000,
            (batch_size, seq_length),
            device=self.device
        )
        attention_mask = torch.ones(
            (batch_size, seq_length),
            device=self.device
        )
        
        # Test training mode
        target_ids = torch.randint(
            0, 1000,
            (batch_size, 128),
            device=self.device
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids
        )
        
        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
        self.assertIn("section_logits", outputs)
        
        # Test inference mode
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        self.assertIn("generated_ids", outputs)
        self.assertIn("section_logits", outputs)
    
    def test_summarize(self):
        """Test text summarization."""
        text = """
        This is a test paper about machine learning.
        It contains multiple sentences and should be summarized.
        The model should generate a coherent summary.
        """
        
        summary = self.model.summarize(text)
        
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
    
    def test_citation_processing(self):
        """Test citation processing."""
        citations = [
            {"text": "Author et al., 2023", "context": "Important finding"}
        ]
        
        # Process citations
        citation_features = self.model._process_citations(citations)
        
        self.assertIsInstance(citation_features, torch.Tensor)
        self.assertEqual(
            citation_features.shape,
            (1, self.model.max_length, 768)
        )
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test empty input
        with self.assertRaises(ValueError):
            self.model.summarize("")
        
        # Test None input
        with self.assertRaises(TypeError):
            self.model.summarize(None)
        
        # Test invalid citations
        with self.assertRaises(TypeError):
            self.model.summarize("Test text", citations="invalid")

class TestCitationProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.processor = CitationProcessor()
    
    def test_citation_embedding(self):
        """Test citation embedding."""
        batch_size = 2
        seq_length = 512
        hidden_size = 768
        
        citations = torch.randn(batch_size, seq_length, hidden_size)
        embeddings = self.processor(citations)
        
        self.assertEqual(embeddings.shape, citations.shape)

class TestSectionClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = SectionClassifier()
    
    def test_section_classification(self):
        """Test section classification."""
        batch_size = 2
        seq_length = 512
        hidden_size = 768
        
        encoded_text = torch.randn(batch_size, seq_length, hidden_size)
        logits = self.classifier(encoded_text)
        
        self.assertEqual(logits.shape, (batch_size, seq_length, 5))

if __name__ == "__main__":
    unittest.main() 