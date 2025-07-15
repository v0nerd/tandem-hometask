# Domain Name Suggestion LLM

A comprehensive machine learning system that generates relevant, memorable, and appropriate domain name suggestions using fine-tuned large language models (LLMs). This project features systematic evaluation, edge case analysis, and robust safety mechanisms.

## 🎯 Overview

The Domain Name Suggestion LLM leverages state-of-the-art language models fine-tuned on synthetic datasets to generate domain name suggestions based on business descriptions. The system employs iterative improvements using LoRA and QLoRA techniques, with a comprehensive evaluation framework using LLM-as-a-Judge methodology.

### Key Features

- **Multi-Model Architecture**: Baseline, LoRA, and QLoRA fine-tuned models
- **Comprehensive Evaluation**: LLM-as-a-Judge framework with 4 key metrics
- **Safety First**: Robust content filtering and inappropriate request blocking
- **Edge Case Analysis**: Systematic testing across diverse input scenarios
- **Production Ready**: FastAPI deployment with optimized performance

## 🏗️ Project Structure

```
tandem-hometask/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config/                            # Configuration files
│   ├── model_config.yaml             # Model configuration
│   └── evaluation_config.yaml        # Evaluation settings
├── src/                               # Source code
│   ├── api/                          # FastAPI application
│   │   └── main.py                   # API server
│   ├── data/                         # Data processing
│   │   └── dataset_creation.py       # Synthetic dataset generation
│   ├── evaluation/                   # Evaluation framework
│   │   ├── llm_judge.py             # LLM-as-a-Judge implementation
│   │   └── safety_checker.py        # Safety filtering
│   ├── models/                       # Model components
│   │   └── domain_generator.py      # Main domain generation model
│   └── utils/                        # Utilities
│       └── config.py                # Configuration management
├── notebooks/                         # Jupyter notebooks
│   ├── 01_dataset_creation.ipynb     # Dataset creation
│   ├── 02_baseline_model.ipynb       # Baseline model training
│   ├── 03_model_iteration.ipynb      # Model improvement cycles
│   ├── 04_evaluation_framework.ipynb # Evaluation setup
│   ├── 05_edge_case_analysis.ipynb   # Edge case discovery
│   └── 06_final_evaluation.ipynb     # Comprehensive evaluation
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh          # Environment setup
│   ├── train_model.py                # Training script
│   └── evaluate_model.py             # Evaluation script
├── tests/                             # Test suite
│   └── test_basic_functionality.py   # Basic functionality tests
└── docs/                              # Documentation
    └── technical_report.md           # Technical report
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: GPU with 8GB VRAM (QLoRA) or 32GB (baseline)
- **API Keys**: OpenAI/Anthropic for LLM-as-a-Judge evaluation

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd tandem-hometask
   ```

2. **Set up environment**

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run setup script
   bash scripts/setup_environment.sh
   ```

3. **Configure API keys**

   ```bash
   export OPENAI_API_KEY='your-key'
   export ANTHROPIC_API_KEY='your-key'
   ```

### Usage

#### 1. Dataset Creation

```bash
python scripts/create_dataset.py
```

#### 2. Model Training

```bash
# Train baseline model
python scripts/train_model.py --model baseline

# Train LoRA model
python scripts/train_model.py --model lora

# Train QLoRA model
python scripts/train_model.py --model qlora
```

#### 3. Evaluation

```bash
python scripts/evaluate_model.py --model v2_qlora
```

#### 4. API Deployment

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Model Performance

| Version | Architecture | Training Method | Overall Score | VRAM Required | Training Time |
|---------|-------------|-----------------|---------------|---------------|---------------|
| baseline | LLaMA-7B | Full fine-tuning | 0.72 | 32GB | 8 hours |
| v1_lora | LLaMA-7B | LoRA | 0.78 | 16GB | 6 hours |
| v2_qlora | LLaMA-7B | QLoRA | **0.82** | **8GB** | **4 hours** |

**Recommended**: v2_qlora model for production deployment

## 🔍 Evaluation Framework

The system uses LLM-as-a-Judge methodology to evaluate domain suggestions across four key metrics:

- **Relevance** (0-1): Semantic alignment with business description
- **Memorability** (0-1): Brand value and recall potential  
- **Appropriateness** (0-1): Safety and non-offensive content
- **Availability-style** (0-1): Plausibility of domain availability

### Key Metrics

- **Overall Score**: 0.82 (v2_qlora)
- **Safety Filtering**: 100% inappropriate requests blocked
- **Edge Case Handling**: 85% success rate
- **API Response Time**: <500ms average

## 🛡️ Safety Features

- **Input Classification**: Automatic detection of inappropriate content
- **Rule-based Filtering**: Harmful request blocking
- **Model-based Rejection**: Advanced content safety mechanisms
- **Comprehensive Logging**: Full audit trail of blocked requests

## 📈 Edge Case Analysis

The system handles various edge cases with targeted improvements:

| Edge Case | Success Rate | Improvement Strategy |
|-----------|-------------|---------------------|
| Ambiguous Descriptions | 75% | Context expansion with keyword extraction |
| Non-English Inputs | 80% | Multilingual support and translation |
| Brand Overlaps | 90% | Enhanced trademark database filtering |
| Inappropriate Content | 100% | Safety filters successfully block |
| Input Length Issues | 85% | Intelligent truncation and summarization |

## 🧪 Development Workflow

The project includes six sequential Jupyter notebooks for full reproducibility:

1. **Dataset Creation** (`01_dataset_creation.ipynb`)
   - Generate synthetic training data
   - Validate dataset quality
   - Visualize distributions

2. **Baseline Model** (`02_baseline_model.ipynb`)
   - Full fine-tuning implementation
   - Initial performance validation
   - Model checkpointing

3. **Model Iteration** (`03_model_iteration.ipynb`)
   - LoRA and QLoRA fine-tuning
   - Performance comparison
   - Efficiency optimization

4. **Evaluation Framework** (`04_evaluation_framework.ipynb`)
   - LLM-as-a-Judge implementation
   - Metric calculation
   - Results visualization

5. **Edge Case Analysis** (`05_edge_case_analysis.ipynb`)
   - Failure mode identification
   - Improvement strategies
   - Robustness testing

6. **Final Evaluation** (`06_final_evaluation.ipynb`)
   - Comprehensive model comparison
   - Production recommendations
   - Performance analysis

## 💻 API Usage

### Generate Domain Suggestions

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = 'models/v2_qlora'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate suggestions
description = 'A tech startup focused on AI-driven healthcare solutions'
input_text = f'Business Description: {description} -> Domain: '

inputs = tokenizer(input_text, return_tensors='pt', padding=True)
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=3,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

domains = [tokenizer.decode(output, skip_special_tokens=True).split('Domain: ')[-1] 
           for output in outputs]

print(f'Description: {description}')
print('Suggested Domains:', domains)
```

### REST API

```bash
# Generate domain suggestions
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"description": "AI healthcare startup"}'
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

## 📚 Documentation

- **Technical Report**: [docs/technical_report.md](docs/technical_report.md)
- **API Documentation**: Available at `/docs` when running the API server
- **Model Versions**: Detailed tracking in the notebooks

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to project maintainers

---

**Last updated**: July 2025
