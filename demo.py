#!/usr/bin/env python3
"""
Domain Name Suggestion LLM - Demo Script

This script demonstrates the key features of the domain name suggestion system
including dataset creation, safety checking, and domain generation.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_dataset_creation():
    """Demonstrate dataset creation functionality."""
    print("=" * 60)
    print("DATASET CREATION DEMO")
    print("=" * 60)
    
    try:
        from data.dataset_creation import SyntheticDatasetCreator
        
        # Initialize creator
        creator = SyntheticDatasetCreator()
        
        # Generate sample business descriptions
        print("\n📝 Generating sample business descriptions...")
        for i in range(3):
            desc = creator.generate_business_description()
            print(f"\n{i+1}. {desc.description}")
            print(f"   Industry: {desc.industry}")
            print(f"   Business Type: {desc.business_type}")
            print(f"   Tone: {desc.tone}")
            print(f"   Keywords: {desc.keywords[:5]}")
        
        # Generate domain suggestions
        print("\n🌐 Generating domain suggestions...")
        desc = creator.generate_business_description()
        suggestions = creator.generate_domain_suggestions(desc, num_suggestions=3)
        
        print(f"\nBusiness: {desc.description}")
        print("Suggested domains:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion.domain} (confidence: {suggestion.confidence:.2f})")
            print(f"     Reasoning: {suggestion.reasoning}")
        
        print("\n✅ Dataset creation demo completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")


def demo_safety_checker():
    """Demonstrate safety checking functionality."""
    print("\n" + "=" * 60)
    print("SAFETY CHECKER DEMO")
    print("=" * 60)
    
    try:
        from evaluation.safety_checker import SafetyChecker
        
        # Initialize safety checker
        checker = SafetyChecker()
        
        # Test cases
        test_cases = [
            ("Professional consulting firm specializing in business optimization", "Safe business"),
            ("Tech startup focused on AI solutions", "Safe tech business"),
            ("Adult content website with explicit material", "Inappropriate content"),
            ("Violence and hate speech platform", "Harmful content"),
            ("A business that helps people", "Ambiguous description"),
            ("Restaurante mexicano en el centro", "Non-English content")
        ]
        
        print("\n🔍 Testing safety checker...")
        for description, label in test_cases:
            result = checker.check_safety(description)
            
            status = "✅ SAFE" if result.is_safe else "❌ BLOCKED"
            print(f"\n{label}:")
            print(f"  Description: {description}")
            print(f"  Status: {status}")
            print(f"  Risk Level: {result.risk_level}")
            print(f"  Confidence: {result.confidence:.2f}")
            
            if result.blocked_reasons:
                print(f"  Blocked Reasons: {', '.join(result.blocked_reasons)}")
        
        print("\n✅ Safety checker demo completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("CONFIGURATION DEMO")
    print("=" * 60)
    
    try:
        from utils.config import load_config, get_config_value
        
        # Load configuration
        config_path = "config/model_config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            
            print(f"\n📋 Configuration loaded from: {config_path}")
            
            # Show some key configuration values
            base_model = get_config_value(config, "base_model.name", "Not found")
            print(f"Base Model: {base_model}")
            
            versions = list(config.get("versions", {}).keys())
            print(f"Available Model Versions: {', '.join(versions)}")
            
            # Show generation settings
            max_length = get_config_value(config, "generation.max_length", "Not found")
            temperature = get_config_value(config, "generation.temperature", "Not found")
            print(f"Generation Max Length: {max_length}")
            print(f"Generation Temperature: {temperature}")
            
            # Show safety settings
            blocked_keywords = config.get("safety", {}).get("blocked_keywords", [])
            print(f"Blocked Keywords: {len(blocked_keywords)} keywords")
            
        else:
            print(f"⚠️ Configuration file not found: {config_path}")
            print("Please run the setup script first")
        
        print("\n✅ Configuration demo completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")


def demo_api_endpoints():
    """Demonstrate API endpoint structure."""
    print("\n" + "=" * 60)
    print("API ENDPOINTS DEMO")
    print("=" * 60)
    
    print("\n🌐 Available API Endpoints:")
    print("  GET  /                    - API information")
    print("  GET  /health              - Health check")
    print("  POST /suggest             - Generate domain suggestions")
    print("  POST /suggest/batch       - Batch domain generation")
    print("  GET  /safety/check        - Safety check")
    print("  GET  /models/info         - Model information")
    
    print("\n📝 Example API Request:")
    print("""
POST /suggest
Content-Type: application/json

{
  "business_description": "organic coffee shop in downtown area",
  "num_suggestions": 3,
  "temperature": 0.7
}
""")
    
    print("📝 Example API Response:")
    print("""
{
  "suggestions": [
    {
      "domain": "organicbeanscafe.com",
      "confidence": 0.92,
      "reasoning": "Direct keyword usage: organic + cafe",
      "tld": ".com",
      "metadata": {"generation_method": "v2_qlora"}
    }
  ],
  "status": "success",
  "metadata": {
    "safety_check": {"is_safe": true, "risk_level": "low"},
    "generation": {"model_version": "v2_qlora", "num_generated": 1}
  }
}
""")
    
    print("\n✅ API endpoints demo completed!")


def demo_project_structure():
    """Show the project structure."""
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    
    structure = """
📁 Domain Name Suggestion LLM/
├── 📄 README.md                    # Project overview and setup
├── 📄 requirements.txt             # Python dependencies
├── 📄 demo.py                      # This demo script
├── 📁 config/                      # Configuration files
│   ├── model_config.yaml          # Model training configuration
│   └── evaluation_config.yaml     # Evaluation framework config
├── 📁 src/                         # Source code
│   ├── data/                       # Data processing modules
│   │   ├── dataset_creation.py    # Synthetic dataset generation
│   │   └── data_loader.py         # Data loading utilities
│   ├── models/                     # Model-related modules
│   │   ├── domain_generator.py    # Main domain generation model
│   │   ├── fine_tuning.py         # Fine-tuning utilities
│   │   └── model_utils.py         # Model helper functions
│   ├── evaluation/                 # Evaluation framework
│   │   ├── llm_judge.py           # LLM-as-a-Judge implementation
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── safety_checker.py      # Safety and filtering
│   ├── api/                        # API deployment
│   │   ├── main.py                # FastAPI application
│   │   └── schemas.py             # API schemas
│   └── utils/                      # Utility functions
│       ├── config.py              # Configuration management
│       └── logging.py             # Logging utilities
├── 📁 notebooks/                   # Jupyter notebooks
│   ├── 01_dataset_creation.ipynb  # Dataset creation experiments
│   ├── 02_baseline_model.ipynb    # Baseline model training
│   ├── 03_model_iteration.ipynb   # Model improvement cycles
│   ├── 04_evaluation_framework.ipynb # Evaluation setup
│   ├── 05_edge_case_analysis.ipynb # Edge case discovery
│   └── 06_final_evaluation.ipynb  # Comprehensive evaluation
├── 📁 scripts/                     # Utility scripts
│   ├── setup_environment.sh       # Environment setup
│   ├── train_model.py             # Training script
│   ├── evaluate_model.py          # Evaluation script
│   └── create_dataset.py          # Dataset creation script
├── 📁 tests/                       # Test suite
│   ├── test_data_creation.py      # Data creation tests
│   ├── test_model.py              # Model tests
│   └── test_evaluation.py         # Evaluation tests
├── 📁 docs/                        # Documentation
│   ├── technical_report.md        # Technical report
│   ├── api_documentation.md       # API documentation
│   └── model_versions.md          # Model version tracking
├── 📁 data/                        # Data directory
│   ├── synthetic/                 # Synthetic training data
│   ├── evaluation/                # Evaluation datasets
│   └── edge_cases/                # Edge case examples
├── 📁 models/                      # Model checkpoints
│   ├── baseline/                  # Baseline model
│   ├── v1_lora/                   # LoRA fine-tuned model
│   └── v2_qlora/                  # QLoRA fine-tuned model
└── 📁 logs/                        # Log files
"""
    
    print(structure)


def demo_usage_instructions():
    """Show usage instructions."""
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    
    instructions = """
🚀 Quick Start:

1. Setup Environment:
   bash scripts/setup_environment.sh

2. Create Dataset:
   python scripts/create_dataset.py --num-samples 1000

3. Train Model:
   python scripts/train_model.py --model v2_qlora

4. Evaluate Model:
   python scripts/evaluate_model.py --model v2_qlora

5. Start API:
   uvicorn src.api.main:app --reload

🔧 Key Features:

• Synthetic Dataset Creation: Generate diverse training data
• Multiple Fine-tuning Methods: Full, LoRA, QLoRA
• LLM-as-a-Judge Evaluation: Comprehensive model assessment
• Safety Framework: Multi-layered content filtering
• REST API: Production-ready deployment
• Comprehensive Testing: Unit and integration tests

📊 Model Versions:

• baseline: Full fine-tuning (0.72 score)
• v1_lora: LoRA fine-tuning (0.78 score)
• v2_qlora: QLoRA fine-tuning (0.82 score) - RECOMMENDED

🛡️ Safety Features:

• Input classification and filtering
• Rule-based rejection mechanisms
• Model-based content analysis
• Comprehensive logging and monitoring

📈 Evaluation Metrics:

• Relevance (30%): Business description alignment
• Memorability (25%): Brand value and recall
• Appropriateness (25%): Safety and compliance
• Availability-style (20%): Domain plausibility
"""
    
    print(instructions)


def main():
    """Main demo function."""
    print("🎯 Domain Name Suggestion LLM - Demo")
    print("A comprehensive AI-powered domain name generation system")
    print("with safety filtering and systematic evaluation")
    
    # Run demos
    demo_project_structure()
    demo_dataset_creation()
    demo_safety_checker()
    demo_configuration()
    demo_api_endpoints()
    demo_usage_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("1. Read the README.md for detailed setup instructions")
    print("2. Check docs/technical_report.md for comprehensive analysis")
    print("3. Run the setup script: bash scripts/setup_environment.sh")
    print("4. Start with dataset creation: python scripts/create_dataset.py")
    print("5. Explore the Jupyter notebooks for experiments")
    print("\n🔗 For more information:")
    print("• Technical Report: docs/technical_report.md")
    print("• API Documentation: docs/api_documentation.md")
    print("• Model Versions: docs/model_versions.md")


if __name__ == "__main__":
    main() 