#!/usr/bin/env python3
"""
Training Script for Domain Name Suggestion LLM

This script handles training of different model versions (baseline, LoRA, QLoRA)
with comprehensive logging, checkpointing, and evaluation.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.domain_generator import DomainGenerator
from utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_model(model_version: str, 
                dataset_path: str,
                output_dir: Optional[str] = None,
                config_path: str = "config/model_config.yaml",
                resume_from_checkpoint: Optional[str] = None) -> None:
    """
    Train a domain name generation model.
    
    Args:
        model_version: Version of model to train (baseline, v1_lora, v2_qlora)
        dataset_path: Path to training dataset
        output_dir: Output directory for model checkpoints
        config_path: Path to model configuration file
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    
    logger.info(f"🚀 Starting training for model version: {model_version}")
    logger.info(f"📊 Dataset: {dataset_path}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Validate inputs
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    if model_version not in config['versions']:
        raise ValueError(f"Unknown model version: {model_version}. Available: {list(config['versions'].keys())}")
    
    # Set output directory
    if output_dir is None:
        output_dir = f"models/{model_version}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize domain generator
        logger.info("🔧 Initializing domain generator...")
        generator = DomainGenerator(config_path, model_version)
        
        # Train the model
        logger.info("🏋️ Starting training...")
        start_time = time.time()
        
        trainer = generator.train(
            dataset_path=dataset_path,
            output_dir=output_dir
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save training metadata
        training_metadata = {
            "model_version": model_version,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "training_time_seconds": training_time,
            "config": config['versions'][model_version],
            "timestamp": time.time()
        }
        
        metadata_path = Path(output_dir) / "training_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        logger.info(f"📝 Training metadata saved to {metadata_path}")
        
        # Optional: Run quick evaluation
        if os.path.exists("data/evaluation/test_set.json"):
            logger.info("🔍 Running quick evaluation...")
            try:
                results = generator.evaluate_model("data/evaluation/test_set.json")
                logger.info(f"📊 Evaluation results: {results['metrics']}")
                
                # Save evaluation results
                eval_path = Path(output_dir) / "evaluation_results.json"
                with open(eval_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"📊 Evaluation results saved to {eval_path}")
                
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
        
        logger.info(f"🎉 Training completed successfully! Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(description="Train Domain Name Suggestion LLM")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="v2_qlora",
        choices=["baseline", "v1_lora", "v2_qlora"],
        help="Model version to train"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/synthetic/training_data.json",
        help="Path to training dataset"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory for model checkpoints"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform a dry run without actual training"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        logger.info("Please run dataset creation first: python scripts/create_dataset.py")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    # Check for required environment variables
    required_env_vars = ["HUGGINGFACE_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("These may be required for model training. Check .env file.")
    
    # Dry run
    if args.dry_run:
        logger.info("🔍 Dry run mode - checking configuration...")
        try:
            config = load_config(args.config)
            logger.info(f"✅ Config loaded successfully")
            logger.info(f"📊 Model version: {args.model}")
            logger.info(f"📁 Dataset: {args.dataset}")
            logger.info(f"📁 Output: {args.output or f'models/{args.model}'}")
            logger.info("✅ Dry run completed - configuration looks good!")
            return
        
        except Exception as e:
            logger.error(f"❌ Dry run failed: {e}")
            sys.exit(1)
    
    # Start training
    try:
        train_model(
            model_version=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            config_path=args.config,
            resume_from_checkpoint=args.resume
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 