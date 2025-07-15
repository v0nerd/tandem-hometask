"""
LLM-as-a-Judge Evaluation Framework

This module implements an automated evaluation system that uses external LLMs
to score domain name suggestions based on multiple criteria:
- Relevance to business description
- Memorability and brand value
- Appropriateness and safety
- Availability-style plausibility
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time
import random

import openai
import anthropic
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Represents the result of a single evaluation."""
    business_description: str
    suggested_domain: str
    metric: str
    score: float
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-a-Judge."""
    provider: str  # "openai", "anthropic", "local"
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    api_key: Optional[str] = None


class LLMJudge:
    """LLM-as-a-Judge implementation for evaluating domain suggestions."""
    
    def __init__(self, config_path: str = "config/evaluation_config.yaml"):
        """Initialize the LLM judge."""
        self.config = self._load_config(config_path)
        self.judge_configs = self._setup_judge_configs()
        self.prompts = self.config['prompts']
        self.weights = self.config['metrics']['weights']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load evaluation configuration."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_judge_configs(self) -> Dict[str, JudgeConfig]:
        """Setup judge configurations for different providers."""
        configs = {}
        
        # Primary judge (OpenAI)
        primary = self.config['llm_judge']['primary']
        configs['primary'] = JudgeConfig(
            provider=primary['provider'],
            model=primary['model'],
            temperature=primary['temperature'],
            max_tokens=primary['max_tokens'],
            timeout=primary['timeout'],
            api_key=self._get_api_key('openai')
        )
        
        # Backup judge (Anthropic)
        backup = self.config['llm_judge']['backup']
        configs['backup'] = JudgeConfig(
            provider=backup['provider'],
            model=backup['model'],
            temperature=backup['temperature'],
            max_tokens=backup['max_tokens'],
            timeout=backup['timeout'],
            api_key=self._get_api_key('anthropic')
        )
        
        return configs
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        if provider == 'openai':
            return os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            return os.getenv('ANTHROPIC_API_KEY')
        return None
    
    async def evaluate_single(self, business_description: str, suggested_domain: str,
                            metric: str, judge_name: str = "primary") -> EvaluationResult:
        """Evaluate a single domain suggestion on a specific metric."""
        
        judge_config = self.judge_configs[judge_name]
        
        # Get prompt for the metric
        prompt_config = self.prompts[metric]
        system_prompt = prompt_config['system']
        user_prompt = prompt_config['user_template'].format(
            business_description=business_description,
            suggested_domain=suggested_domain
        )
        
        try:
            # Call the appropriate LLM
            if judge_config.provider == "openai":
                response = await self._call_openai(judge_config, system_prompt, user_prompt)
            elif judge_config.provider == "anthropic":
                response = await self._call_anthropic(judge_config, system_prompt, user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {judge_config.provider}")
            
            # Parse the response
            result = self._parse_evaluation_response(response, business_description, 
                                                   suggested_domain, metric)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {metric} for {suggested_domain}: {e}")
            # Return a default low score on error
            return EvaluationResult(
                business_description=business_description,
                suggested_domain=suggested_domain,
                metric=metric,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "judge": judge_name}
            )
    
    async def _call_openai(self, judge_config: JudgeConfig, system_prompt: str, 
                          user_prompt: str) -> str:
        """Call OpenAI API for evaluation."""
        client = openai.AsyncOpenAI(api_key=judge_config.api_key)
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=judge_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=judge_config.temperature,
                max_tokens=judge_config.max_tokens
            ),
            timeout=judge_config.timeout
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(self, judge_config: JudgeConfig, system_prompt: str,
                            user_prompt: str) -> str:
        """Call Anthropic API for evaluation."""
        client = anthropic.AsyncAnthropic(api_key=judge_config.api_key)
        
        response = await asyncio.wait_for(
            client.messages.create(
                model=judge_config.model,
                max_tokens=judge_config.max_tokens,
                temperature=judge_config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            ),
            timeout=judge_config.timeout
        )
        
        return response.content[0].text
    
    def _parse_evaluation_response(self, response: str, business_description: str,
                                 suggested_domain: str, metric: str) -> EvaluationResult:
        """Parse the LLM response into an EvaluationResult."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                score = float(parsed.get('score', 0.0))
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                
                # Validate score range
                score = max(0.0, min(1.0, score))
                
                return EvaluationResult(
                    business_description=business_description,
                    suggested_domain=suggested_domain,
                    metric=metric,
                    score=score,
                    reasoning=reasoning,
                    confidence=0.9,  # High confidence for structured response
                    metadata={"raw_response": response}
                )
            else:
                # Fallback: try to extract score from text
                score = self._extract_score_from_text(response)
                return EvaluationResult(
                    business_description=business_description,
                    suggested_domain=suggested_domain,
                    metric=metric,
                    score=score,
                    reasoning=response,
                    confidence=0.5,  # Lower confidence for unstructured response
                    metadata={"raw_response": response, "parsing_method": "text_extraction"}
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback: try to extract score from text
            score = self._extract_score_from_text(response)
            return EvaluationResult(
                business_description=business_description,
                suggested_domain=suggested_domain,
                metric=metric,
                score=score,
                reasoning=response,
                confidence=0.3,  # Low confidence for failed parsing
                metadata={"raw_response": response, "parsing_error": str(e)}
            )
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract a score from unstructured text."""
        import re
        
        # Look for patterns like "score: 0.8" or "0.8/1.0"
        score_patterns = [
            r'score[:\s]*(\d+\.?\d*)',
            r'(\d+\.?\d*)/1\.0',
            r'(\d+\.?\d*)\s*out\s*of\s*1',
            r'rating[:\s]*(\d+\.?\d*)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        # If no score found, try to infer from sentiment
        positive_words = ['excellent', 'great', 'good', 'high', 'strong', 'perfect']
        negative_words = ['poor', 'bad', 'low', 'weak', 'terrible', 'inappropriate']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 0.7
        elif negative_count > positive_count:
            return 0.3
        else:
            return 0.5
    
    async def evaluate_comprehensive(self, business_description: str, 
                                   suggested_domains: List[str],
                                   metrics: Optional[List[str]] = None) -> List[Dict]:
        """Evaluate multiple domain suggestions on all metrics."""
        
        if metrics is None:
            metrics = list(self.prompts.keys())
        
        results = []
        
        for domain in suggested_domains:
            domain_results = {}
            
            # Evaluate on each metric
            for metric in metrics:
                try:
                    result = await self.evaluate_single(
                        business_description, domain, metric
                    )
                    domain_results[metric] = result
                    
                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {metric} for {domain}: {e}")
                    domain_results[metric] = EvaluationResult(
                        business_description=business_description,
                        suggested_domain=domain,
                        metric=metric,
                        score=0.0,
                        reasoning=f"Evaluation failed: {str(e)}",
                        confidence=0.0,
                        metadata={"error": str(e)}
                    )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(domain_results)
            
            results.append({
                "domain": domain,
                "overall_score": overall_score,
                "metric_scores": {metric: result.score for metric, result in domain_results.items()},
                "evaluations": domain_results
            })
        
        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return results
    
    def _calculate_overall_score(self, domain_results: Dict[str, EvaluationResult]) -> float:
        """Calculate overall score using weighted average."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, result in domain_results.items():
            weight = self.weights.get(metric, 0.25)  # Default weight
            total_score += result.score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    async def evaluate_dataset(self, dataset_path: str, 
                             output_path: str,
                             max_samples: Optional[int] = None) -> Dict:
        """Evaluate an entire dataset of domain suggestions."""
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        logger.info(f"Evaluating {len(dataset)} samples")
        
        all_results = []
        summary_stats = {
            "total_samples": len(dataset),
            "metrics": {},
            "overall_scores": []
        }
        
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
            try:
                business_desc = sample['input']
                suggested_domains = [sample['output']]  # Single domain per sample
                
                results = await self.evaluate_comprehensive(
                    business_desc, suggested_domains
                )
                
                all_results.append({
                    "sample_id": i,
                    "business_description": business_desc,
                    "results": results
                })
                
                # Update summary stats
                if results:
                    overall_score = results[0]['overall_score']
                    summary_stats['overall_scores'].append(overall_score)
                    
                    # Update metric stats
                    for metric, score in results[0]['metric_scores'].items():
                        if metric not in summary_stats['metrics']:
                            summary_stats['metrics'][metric] = []
                        summary_stats['metrics'][metric].append(score)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
        
        # Calculate final statistics
        final_summary = self._calculate_summary_statistics(summary_stats)
        
        # Save results
        output_data = {
            "summary": final_summary,
            "detailed_results": all_results,
            "metadata": {
                "evaluation_timestamp": time.time(),
                "num_samples": len(all_results),
                "metrics_evaluated": list(self.prompts.keys())
            }
        }
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        logger.info(f"Average overall score: {final_summary['overall']['mean']:.3f}")
        
        return output_data
    
    def _calculate_summary_statistics(self, summary_stats: Dict) -> Dict:
        """Calculate comprehensive summary statistics."""
        import numpy as np
        
        final_summary = {
            "overall": {},
            "metrics": {}
        }
        
        # Overall score statistics
        if summary_stats['overall_scores']:
            scores = np.array(summary_stats['overall_scores'])
            final_summary['overall'] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "percentiles": {
                    "25": float(np.percentile(scores, 25)),
                    "75": float(np.percentile(scores, 75)),
                    "90": float(np.percentile(scores, 90))
                }
            }
        
        # Metric-specific statistics
        for metric, scores in summary_stats['metrics'].items():
            if scores:
                scores_array = np.array(scores)
                final_summary['metrics'][metric] = {
                    "mean": float(np.mean(scores_array)),
                    "std": float(np.std(scores_array)),
                    "median": float(np.median(scores_array)),
                    "min": float(np.min(scores_array)),
                    "max": float(np.max(scores_array))
                }
        
        return final_summary


async def main():
    """Main function for testing the LLM judge."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset to evaluate")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--config", type=str, default="config/evaluation_config.yaml",
                       help="Path to evaluation config")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    judge = LLMJudge(args.config)
    await judge.evaluate_dataset(args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    asyncio.run(main()) 