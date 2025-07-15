"""
Domain Name Generator Model

This module implements the main domain name generation model with support for:
- Multiple fine-tuning methods (full, LoRA, QLoRA)
- Model versioning and checkpointing
- Inference with safety filtering
- Integration with evaluation framework
"""

import json
import logging
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import time

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np

from ..evaluation.safety_checker import SafetyChecker
from ..utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for domain generation."""
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 3
    pad_token_id: int = 0
    eos_token_id: int = 2


@dataclass
class DomainSuggestion:
    """Represents a generated domain suggestion."""
    domain: str
    confidence: float
    reasoning: str
    tld: str
    metadata: Dict[str, Any]


class DomainGenerator:
    """Main domain name generation model."""
    
    def __init__(self, config_path: str = "config/model_config.yaml", 
                 model_version: str = "v2_qlora"):
        """Initialize the domain generator."""
        self.config = load_config(config_path)
        self.model_version = model_version
        self.version_config = self.config['versions'][model_version]
        
        self.tokenizer = None
        self.model = None
        self.safety_checker = SafetyChecker()
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer based on version."""
        base_model_name = self.config['base_model']['name']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=self.config['base_model']['trust_remote_code'],
            use_auth_token=self.config['base_model']['use_auth_token']
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=self.config['base_model']['trust_remote_code'],
            use_auth_token=self.config['base_model']['use_auth_token'],
            torch_dtype=torch.bfloat16 if self.config['hardware']['mixed_precision'] == 'bf16' else torch.float16,
            device_map="auto" if self.config['hardware']['device'] == 'auto' else None
        )
        
        # Apply fine-tuning method
        training_method = self.version_config['training_method']
        if training_method == "lora":
            self._apply_lora()
        elif training_method == "qlora":
            self._apply_qlora()
        
        # Load fine-tuned weights if available
        self._load_fine_tuned_weights()
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _apply_lora(self):
        """Apply LoRA fine-tuning configuration."""
        lora_config = self.version_config['lora_config']
        
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, peft_config)
        logger.info("Applied LoRA configuration")
        
    def _apply_qlora(self):
        """Apply QLoRA fine-tuning configuration."""
        qlora_config = self.version_config['qlora_config']
        
        # QLoRA is similar to LoRA but with quantization
        peft_config = LoraConfig(
            r=qlora_config['r'],
            lora_alpha=qlora_config['lora_alpha'],
            target_modules=qlora_config['target_modules'],
            lora_dropout=qlora_config['lora_dropout'],
            bias=qlora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, peft_config)
        logger.info("Applied QLoRA configuration")
        
    def _load_fine_tuned_weights(self):
        """Load fine-tuned weights if available."""
        model_path = f"models/{self.model_version}"
        
        if Path(model_path).exists():
            try:
                # Load the fine-tuned model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16 if self.config['hardware']['mixed_precision'] == 'bf16' else torch.float16,
                    device_map="auto" if self.config['hardware']['device'] == 'auto' else None
                )
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned weights: {e}")
        else:
            logger.info(f"No fine-tuned weights found at {model_path}, using base model")
    
    def prepare_training_data(self, dataset_path: str) -> Dataset:
        """Prepare training data from the synthetic dataset."""
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Format data for training
        formatted_data = []
        for item in raw_data:
            # Create training prompt
            prompt = self._create_training_prompt(item['input'], item['output'])
            formatted_data.append({
                'text': prompt,
                'input_length': len(item['input']),
                'output_length': len(item['output'])
            })
        
        return Dataset.from_list(formatted_data)
    
    def _create_training_prompt(self, business_description: str, target_domain: str) -> str:
        """Create training prompt in the format expected by the model."""
        prompt = f"Business Description: {business_description}\n"
        prompt += f"Suggested Domain: {target_domain}\n"
        prompt += f"<|endoftext|>"
        return prompt
    
    def train(self, dataset_path: str, output_dir: str = None):
        """Train the model on the provided dataset."""
        if output_dir is None:
            output_dir = f"models/{self.model_version}"
        
        # Prepare data
        dataset = self.prepare_training_data(dataset_path)
        
        # Split dataset
        train_val = dataset.train_test_split(test_size=0.1)
        train_dataset = train_val['train']
        val_dataset = train_val['test']
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.version_config['learning_rate'],
            per_device_train_batch_size=self.version_config['batch_size'],
            per_device_eval_batch_size=self.version_config['batch_size'],
            gradient_accumulation_steps=self.version_config['gradient_accumulation_steps'],
            num_train_epochs=self.version_config['num_epochs'],
            warmup_steps=self.version_config['warmup_steps'],
            weight_decay=self.version_config['weight_decay'],
            max_grad_norm=self.version_config['max_grad_norm'],
            save_steps=self.version_config['save_steps'],
            eval_steps=self.version_config['eval_steps'],
            logging_steps=self.version_config['logging_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config['logging']['wandb_project'] else None,
            dataloader_num_workers=self.config['hardware']['dataloader_num_workers'],
            gradient_checkpointing=self.config['hardware']['gradient_checkpointing'],
            fp16=self.config['hardware']['mixed_precision'] == 'fp16',
            bf16=self.config['hardware']['mixed_precision'] == 'bf16',
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info(f"Starting training for model version: {self.model_version}")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        return trainer
    
    def generate_domains(self, business_description: str, 
                        num_suggestions: int = 3,
                        generation_config: Optional[GenerationConfig] = None) -> List[DomainSuggestion]:
        """Generate domain name suggestions for a business description."""
        
        # Safety check
        if not self.safety_checker.is_safe(business_description):
            logger.warning(f"Unsafe business description detected: {business_description}")
            return []
        
        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Create input prompt
        prompt = f"Business Description: {business_description}\n"
        prompt += "Suggested Domain:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=self.config['data']['max_input_length'])
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=generation_config.max_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                do_sample=generation_config.do_sample,
                num_return_sequences=generation_config.num_return_sequences,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                early_stopping=True
            )
        
        # Decode outputs
        suggestions = []
        for output in outputs:
            # Decode the generated text
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract domain from generated text
            domain = self._extract_domain_from_text(generated_text, business_description)
            
            if domain:
                # Calculate confidence based on generation probability
                confidence = self._calculate_confidence(output, inputs)
                
                suggestion = DomainSuggestion(
                    domain=domain,
                    confidence=confidence,
                    reasoning=f"Generated for: {business_description}",
                    tld=self._extract_tld(domain),
                    metadata={
                        "generation_method": self.model_version,
                        "business_description": business_description,
                        "raw_generated_text": generated_text
                    }
                )
                
                suggestions.append(suggestion)
        
        # Remove duplicates and sort by confidence
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion.domain not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion.domain)
        
        # Sort by confidence and limit
        unique_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return unique_suggestions[:num_suggestions]
    
    def _extract_domain_from_text(self, generated_text: str, business_description: str) -> Optional[str]:
        """Extract domain name from generated text."""
        # Look for the domain after "Suggested Domain:"
        if "Suggested Domain:" in generated_text:
            domain_part = generated_text.split("Suggested Domain:")[1].strip()
            # Take the first line or word
            domain = domain_part.split('\n')[0].split()[0].strip()
            
            # Clean up the domain
            domain = domain.lower()
            domain = domain.replace(' ', '')
            
            # Add .com if no TLD
            if '.' not in domain:
                domain += '.com'
            
            # Validate domain format
            if self._is_valid_domain(domain):
                return domain
        
        return None
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain has valid format."""
        import re
        # Basic domain validation
        pattern = r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$'
        return bool(re.match(pattern, domain))
    
    def _extract_tld(self, domain: str) -> str:
        """Extract TLD from domain."""
        if '.' in domain:
            return '.' + domain.split('.')[-1]
        return '.com'
    
    def _calculate_confidence(self, output_tokens: torch.Tensor, 
                            input_tokens: Dict[str, torch.Tensor]) -> float:
        """Calculate confidence score based on generation probability."""
        # Simple confidence calculation based on sequence length
        # In practice, you might want to use perplexity or other metrics
        input_length = input_tokens['input_ids'].shape[1]
        generated_length = output_tokens.shape[1] - input_length
        
        # Normalize by expected length
        expected_length = 20  # Expected domain name length
        length_ratio = min(generated_length / expected_length, 2.0)
        
        # Base confidence with length penalty
        confidence = max(0.1, 1.0 - abs(1.0 - length_ratio) * 0.3)
        
        return confidence
    
    def evaluate_model(self, test_dataset_path: str) -> Dict[str, Any]:
        """Evaluate the model on a test dataset."""
        from ..evaluation.llm_judge import LLMJudge
        
        # Load test data
        with open(test_dataset_path, 'r') as f:
            test_data = json.load(f)
        
        # Generate predictions
        predictions = []
        for sample in test_data[:100]:  # Limit for evaluation
            business_desc = sample['input']
            suggestions = self.generate_domains(business_desc, num_suggestions=1)
            
            if suggestions:
                predictions.append({
                    'business_description': business_desc,
                    'predicted_domain': suggestions[0].domain,
                    'confidence': suggestions[0].confidence,
                    'ground_truth': sample['output']
                })
        
        # Use LLM-as-a-Judge for evaluation
        judge = LLMJudge()
        
        # Evaluate each prediction
        evaluation_results = []
        for pred in predictions:
            results = await judge.evaluate_comprehensive(
                pred['business_description'], 
                [pred['predicted_domain']]
            )
            if results:
                evaluation_results.append(results[0])
        
        # Calculate metrics
        metrics = self._calculate_evaluation_metrics(evaluation_results)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'evaluation_results': evaluation_results
        }
    
    def _calculate_evaluation_metrics(self, evaluation_results: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics from LLM judge results."""
        if not evaluation_results:
            return {}
        
        overall_scores = [result['overall_score'] for result in evaluation_results]
        
        metrics = {
            'mean_overall_score': np.mean(overall_scores),
            'std_overall_score': np.std(overall_scores),
            'median_overall_score': np.median(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'max_overall_score': np.max(overall_scores)
        }
        
        # Calculate metric-specific scores
        for metric in ['relevance', 'memorability', 'appropriateness', 'availability_style']:
            scores = []
            for result in evaluation_results:
                if metric in result['metric_scores']:
                    scores.append(result['metric_scores'][metric])
            
            if scores:
                metrics[f'mean_{metric}_score'] = np.mean(scores)
                metrics[f'std_{metric}_score'] = np.std(scores)
        
        return metrics


def main():
    """Main function for training and evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Name Generator")
    parser.add_argument("--action", choices=["train", "generate", "evaluate"], required=True,
                       help="Action to perform")
    parser.add_argument("--model_version", type=str, default="v2_qlora",
                       help="Model version to use")
    parser.add_argument("--dataset_path", type=str,
                       help="Path to training dataset")
    parser.add_argument("--business_description", type=str,
                       help="Business description for generation")
    parser.add_argument("--test_dataset_path", type=str,
                       help="Path to test dataset for evaluation")
    parser.add_argument("--config_path", type=str, default="config/model_config.yaml",
                       help="Path to model config")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    generator = DomainGenerator(args.config_path, args.model_version)
    
    if args.action == "train":
        if not args.dataset_path:
            raise ValueError("Dataset path required for training")
        generator.train(args.dataset_path)
        
    elif args.action == "generate":
        if not args.business_description:
            raise ValueError("Business description required for generation")
        suggestions = generator.generate_domains(args.business_description)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion.domain} (confidence: {suggestion.confidence:.2f})")
            
    elif args.action == "evaluate":
        if not args.test_dataset_path:
            raise ValueError("Test dataset path required for evaluation")
        results = generator.evaluate_model(args.test_dataset_path)
        print("Evaluation Results:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.3f}")


if __name__ == "__main__":
    main() 