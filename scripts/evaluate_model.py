#!/usr/bin/env python3
"""
Evaluation Script for Domain Name Suggestion LLM

This script runs comprehensive evaluation using the LLM-as-a-Judge framework
to assess model performance across multiple metrics.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.llm_judge import LLMJudge
from models.domain_generator import DomainGenerator
from utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def evaluate_model(model_version: str,
                        test_dataset_path: str,
                        output_path: Optional[str] = None,
                        config_path: str = "config/evaluation_config.yaml",
                        max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate a trained model using LLM-as-a-Judge framework.
    
    Args:
        model_version: Version of model to evaluate
        test_dataset_path: Path to test dataset
        output_path: Path to save evaluation results
        config_path: Path to evaluation configuration
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    
    logger.info(f"🔍 Starting evaluation for model version: {model_version}")
    logger.info(f"📊 Test dataset: {test_dataset_path}")
    logger.info(f"📁 Output path: {output_path}")
    
    # Validate inputs
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set output path
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"evaluation_results_{model_version}_{timestamp}.json"
    
    try:
        # Initialize LLM judge
        logger.info("🔧 Initializing LLM judge...")
        judge = LLMJudge(config_path)
        
        # Run evaluation
        logger.info("🏃 Running evaluation...")
        start_time = time.time()
        
        results = await judge.evaluate_dataset(
            dataset_path=test_dataset_path,
            output_path=output_path,
            max_samples=max_samples
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Print summary
        summary = results.get('summary', {})
        if 'overall' in summary:
            overall_score = summary['overall'].get('mean', 0)
            logger.info(f"📊 Overall Score: {overall_score:.3f}")
        
        # Print metric scores
        if 'metrics' in summary:
            logger.info("📈 Metric Scores:")
            for metric, scores in summary['metrics'].items():
                mean_score = scores.get('mean', 0)
                logger.info(f"  {metric}: {mean_score:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


def compare_models(model_versions: list,
                  test_dataset_path: str,
                  output_dir: str = "evaluation_comparison",
                  config_path: str = "config/evaluation_config.yaml",
                  max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Compare multiple model versions.
    
    Args:
        model_versions: List of model versions to compare
        test_dataset_path: Path to test dataset
        output_dir: Directory to save comparison results
        config_path: Path to evaluation configuration
        max_samples: Maximum number of samples per model
        
    Returns:
        Comparison results dictionary
    """
    
    logger.info(f"🔄 Comparing models: {model_versions}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        "models": {},
        "summary": {},
        "metadata": {
            "test_dataset": test_dataset_path,
            "model_versions": model_versions,
            "timestamp": time.time()
        }
    }
    
    for model_version in model_versions:
        logger.info(f"🔍 Evaluating {model_version}...")
        
        try:
            # Run evaluation for this model
            output_path = os.path.join(output_dir, f"{model_version}_results.json")
            
            # Run evaluation asynchronously
            results = asyncio.run(evaluate_model(
                model_version=model_version,
                test_dataset_path=test_dataset_path,
                output_path=output_path,
                config_path=config_path,
                max_samples=max_samples
            ))
            
            comparison_results["models"][model_version] = results
            
            # Extract summary
            if "summary" in results:
                comparison_results["summary"][model_version] = results["summary"]
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_version}: {e}")
            comparison_results["models"][model_version] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"📊 Comparison results saved to {comparison_path}")
    
    # Print comparison summary
    print_comparison_summary(comparison_results)
    
    return comparison_results


def print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
    """Print a summary of model comparison results."""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    summary = comparison_results.get("summary", {})
    
    # Print overall scores
    print("\n📊 Overall Scores:")
    print("-" * 40)
    for model_version, model_summary in summary.items():
        if "overall" in model_summary:
            overall_score = model_summary["overall"].get("mean", 0)
            print(f"{model_version:15} | {overall_score:.3f}")
    
    # Print metric breakdown
    if summary:
        first_model = list(summary.keys())[0]
        if "metrics" in summary[first_model]:
            print("\n📈 Metric Breakdown:")
            print("-" * 40)
            
            metrics = summary[first_model]["metrics"].keys()
            for metric in metrics:
                print(f"\n{metric.upper()}:")
                for model_version, model_summary in summary.items():
                    if "metrics" in model_summary and metric in model_summary["metrics"]:
                        score = model_summary["metrics"][metric].get("mean", 0)
                        print(f"  {model_version:15} | {score:.3f}")
    
    print("\n" + "="*60)


def generate_evaluation_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate a detailed evaluation report."""
    
    report = {
        "evaluation_report": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": results.get("summary", {}),
            "detailed_results": results.get("detailed_results", []),
            "metadata": results.get("metadata", {})
        }
    }
    
    # Add performance analysis
    if "summary" in results and "overall" in results["summary"]:
        overall = results["summary"]["overall"]
        report["evaluation_report"]["performance_analysis"] = {
            "score_distribution": {
                "excellent": sum(1 for score in results.get("detailed_results", []) 
                               if score.get("overall_score", 0) >= 0.85),
                "good": sum(1 for score in results.get("detailed_results", []) 
                           if 0.7 <= score.get("overall_score", 0) < 0.85),
                "acceptable": sum(1 for score in results.get("detailed_results", []) 
                                if 0.5 <= score.get("overall_score", 0) < 0.7),
                "poor": sum(1 for score in results.get("detailed_results", []) 
                           if score.get("overall_score", 0) < 0.5)
            },
            "statistics": overall
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Evaluation report saved to {output_path}")


async def main():
    """Main function for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Domain Name Suggestion LLM")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="v2_qlora",
        help="Model version to evaluate"
    )
    
    parser.add_argument(
        "--test-dataset", 
        type=str, 
        default="data/evaluation/test_set.json",
        help="Path to test dataset"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path for evaluation results"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/evaluation_config.yaml",
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    parser.add_argument(
        "--compare", 
        nargs='+',
        help="Compare multiple model versions"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_comparison",
        help="Output directory for comparison results"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform a dry run without actual evaluation"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.test_dataset):
        logger.error(f"Test dataset not found: {args.test_dataset}")
        logger.info("Please create a test dataset first")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        logger.error(f"Evaluation config not found: {args.config}")
        sys.exit(1)
    
    # Check for required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("These are required for LLM-as-a-Judge evaluation. Check .env file.")
    
    # Dry run
    if args.dry_run:
        logger.info("🔍 Dry run mode - checking configuration...")
        try:
            config = load_config(args.config)
            logger.info(f"✅ Config loaded successfully")
            logger.info(f"📊 Test dataset: {args.test_dataset}")
            logger.info(f"📁 Output: {args.output or 'evaluation_results.json'}")
            logger.info("✅ Dry run completed - configuration looks good!")
            return
        
        except Exception as e:
            logger.error(f"❌ Dry run failed: {e}")
            sys.exit(1)
    
    # Run evaluation
    try:
        if args.compare:
            # Compare multiple models
            comparison_results = compare_models(
                model_versions=args.compare,
                test_dataset_path=args.test_dataset,
                output_dir=args.output_dir,
                config_path=args.config,
                max_samples=args.max_samples
            )
        else:
            # Evaluate single model
            results = await evaluate_model(
                model_version=args.model,
                test_dataset_path=args.test_dataset,
                output_path=args.output,
                config_path=args.config,
                max_samples=args.max_samples
            )
            
            # Generate report
            if args.output:
                report_path = args.output.replace('.json', '_report.json')
                generate_evaluation_report(results, report_path)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Evaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 