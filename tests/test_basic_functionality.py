"""
Basic functionality tests for Domain Name Suggestion LLM

This module contains basic tests to verify the core functionality
of the domain name suggestion system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset_creation import SyntheticDatasetCreator, BusinessDescription
from evaluation.safety_checker import SafetyChecker
from utils.config import load_config


class TestDatasetCreation(unittest.TestCase):
    """Test dataset creation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.creator = SyntheticDatasetCreator()
    
    def test_business_description_generation(self):
        """Test business description generation."""
        desc = self.creator.generate_business_description()
        
        self.assertIsInstance(desc, BusinessDescription)
        self.assertIsInstance(desc.description, str)
        self.assertGreater(len(desc.description), 10)
        self.assertIn(desc.industry, self.creator.industries)
        self.assertIn(desc.business_type, self.creator.business_types)
        self.assertIn(desc.tone, self.creator.tones)
        self.assertIn(desc.complexity, self.creator.complexities)
    
    def test_domain_suggestion_generation(self):
        """Test domain suggestion generation."""
        desc = self.creator.generate_business_description()
        suggestions = self.creator.generate_domain_suggestions(desc, num_suggestions=3)
        
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 3)
        
        for suggestion in suggestions:
            self.assertIsInstance(suggestion.domain, str)
            self.assertGreater(len(suggestion.domain), 0)
            self.assertIsInstance(suggestion.confidence, float)
            self.assertGreaterEqual(suggestion.confidence, 0.0)
            self.assertLessEqual(suggestion.confidence, 1.0)


class TestSafetyChecker(unittest.TestCase):
    """Test safety checking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = SafetyChecker()
    
    def test_safe_content(self):
        """Test that safe content passes safety checks."""
        safe_description = "Professional consulting firm specializing in business optimization"
        result = self.checker.check_safety(safe_description)
        
        self.assertTrue(result.is_safe)
        self.assertIn(result.risk_level, ["low", "medium", "high"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_unsafe_content(self):
        """Test that unsafe content is blocked."""
        unsafe_description = "Adult content website with explicit material"
        result = self.checker.check_safety(unsafe_description)
        
        self.assertFalse(result.is_safe)
        self.assertEqual(result.risk_level, "blocked")
        self.assertGreater(len(result.blocked_reasons), 0)
    
    def test_edge_case_content(self):
        """Test edge case content handling."""
        edge_cases = [
            "A business that helps people",  # Ambiguous
            "Tech startup",  # Very short
            "Restaurante mexicano",  # Non-English
        ]
        
        for description in edge_cases:
            result = self.checker.check_safety(description)
            self.assertIsInstance(result.is_safe, bool)
            self.assertIsInstance(result.risk_level, str)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def test_config_loading(self):
        """Test configuration file loading."""
        config_path = "config/model_config.yaml"
        
        if os.path.exists(config_path):
            config = load_config(config_path)
            
            self.assertIsInstance(config, dict)
            self.assertIn("base_model", config)
            self.assertIn("versions", config)
            self.assertIn("generation", config)
        else:
            self.skipTest(f"Config file not found: {config_path}")


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_end_to_end_workflow(self):
        """Test basic end-to-end workflow."""
        # Create dataset creator
        creator = SyntheticDatasetCreator()
        
        # Generate business description
        desc = creator.generate_business_description()
        self.assertIsInstance(desc, BusinessDescription)
        
        # Generate domain suggestions
        suggestions = creator.generate_domain_suggestions(desc, num_suggestions=2)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Check safety
        checker = SafetyChecker()
        safety_result = checker.check_safety(desc.description)
        self.assertIsInstance(safety_result.is_safe, bool)


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2) 