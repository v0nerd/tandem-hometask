"""
Synthetic Dataset Creation for Domain Name Suggestion LLM

This module creates synthetic training data for domain name generation by:
1. Generating diverse business descriptions
2. Creating corresponding target domain names
3. Ensuring data quality and diversity
4. Supporting different industries and business types
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import secrets

logger = logging.getLogger(__name__)


@dataclass
class BusinessDescription:
    """Represents a business description with metadata."""
    description: str
    industry: str
    business_type: str
    target_audience: str
    tone: str
    complexity: str
    location: Optional[str] = None
    keywords: Optional[List[str]] = None


@dataclass
class DomainSuggestion:
    """Represents a domain name suggestion."""
    domain: str
    confidence: float
    reasoning: str
    tld: str = ".com"


class SyntheticDatasetCreator:
    """Creates synthetic training data for domain name generation."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the dataset creator."""
        self.config = self._load_config(config_path)
        self.industries = self._get_industries()
        self.business_types = self._get_business_types()
        self.target_audiences = self._get_target_audiences()
        self.tones = self._get_tones()
        self.complexities = self._get_complexities()
        self.tlds = [".com", ".org", ".net", ".io", ".co", ".app", ".tech", ".store"]
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_industries(self) -> List[str]:
        """Get list of industries for data generation."""
        return [
            "technology", "healthcare", "finance", "education", "retail", 
            "food_beverage", "consulting", "non_profit", "real_estate", 
            "entertainment", "fitness", "beauty", "automotive", "travel",
            "manufacturing", "logistics", "legal", "marketing", "design",
            "environmental", "pet_care", "childcare", "senior_care"
        ]
    
    def _get_business_types(self) -> List[str]:
        """Get list of business types."""
        return [
            "startup", "established_company", "freelance", "agency", 
            "consultancy", "non_profit", "cooperative", "franchise",
            "online_business", "brick_mortar", "hybrid"
        ]
    
    def _get_target_audiences(self) -> List[str]:
        """Get list of target audiences."""
        return [
            "consumers", "businesses", "students", "professionals",
            "seniors", "parents", "children", "pet_owners", "travelers",
            "fitness_enthusiasts", "tech_savvy", "budget_conscious"
        ]
    
    def _get_tones(self) -> List[str]:
        """Get list of communication tones."""
        return [
            "professional", "casual", "friendly", "luxury", "budget",
            "innovative", "traditional", "modern", "rustic", "sophisticated"
        ]
    
    def _get_complexities(self) -> List[str]:
        """Get list of complexity levels."""
        return ["simple", "moderate", "complex"]
    
    def generate_business_description(self) -> BusinessDescription:
        """Generate a random business description."""
        industry = secrets.choice(self.industries)
        business_type = secrets.choice(self.business_types)
        target_audience = secrets.choice(self.target_audiences)
        tone = secrets.choice(self.tones)
        complexity = secrets.choice(self.complexities)
        
        description = self._create_description(
            industry, business_type, target_audience, tone, complexity
        )
        
        keywords = self._extract_keywords(description)
        
        return BusinessDescription(
            description=description,
            industry=industry,
            business_type=business_type,
            target_audience=target_audience,
            tone=tone,
            complexity=complexity,
            keywords=keywords
        )
    
    def _create_description(self, industry: str, business_type: str, 
                          target_audience: str, tone: str, complexity: str) -> str:
        """Create a business description based on parameters."""
        
        # Industry-specific templates
        industry_templates = {
            "technology": [
                "{tone} {business_type} specializing in {tech_focus} for {target_audience}",
                "Innovative {business_type} providing {tech_solution} to {target_audience}",
                "{tone} tech {business_type} focused on {tech_area} solutions"
            ],
            "healthcare": [
                "{tone} healthcare {business_type} providing {health_service}",
                "Professional {business_type} specializing in {health_focus}",
                "{tone} medical {business_type} serving {target_audience}"
            ],
            "food_beverage": [
                "{tone} {business_type} serving {food_type} cuisine",
                "Family-owned {business_type} specializing in {food_focus}",
                "{tone} restaurant {business_type} offering {dining_experience}"
            ],
            "consulting": [
                "{tone} consulting {business_type} helping {target_audience} with {consulting_area}",
                "Professional {business_type} providing {consulting_service}",
                "{tone} advisory {business_type} specializing in {business_focus}"
            ]
        }
        
        # Default template
        default_template = "{tone} {business_type} in the {industry} industry serving {target_audience}"
        
        template = industry_templates.get(industry, [default_template])
        template = secrets.choice(template)
        
        # Fill in placeholders
        description = template.format(
            tone=tone,
            business_type=business_type.replace('_', ' '),
            industry=industry.replace('_', ' '),
            target_audience=target_audience.replace('_', ' '),
            tech_focus=secrets.choice(["AI solutions", "web development", "mobile apps", "cloud services"]),
            tech_solution=secrets.choice(["digital transformation", "automation", "data analytics"]),
            tech_area=secrets.choice(["machine learning", "cybersecurity", "IoT", "blockchain"]),
            health_service=secrets.choice(["primary care", "specialized treatment", "preventive care"]),
            health_focus=secrets.choice(["pediatrics", "cardiology", "dermatology", "orthopedics"]),
            food_type=secrets.choice(["Italian", "Mexican", "Asian fusion", "American", "Mediterranean"]),
            food_focus=secrets.choice(["organic ingredients", "farm-to-table", "artisan bread", "craft coffee"]),
            dining_experience=secrets.choice(["casual dining", "fine dining", "fast casual", "food truck"]),
            consulting_area=secrets.choice(["strategy", "operations", "marketing", "finance", "technology"]),
            consulting_service=secrets.choice(["business optimization", "growth strategy", "process improvement"]),
            business_focus=secrets.choice(["startup consulting", "enterprise solutions", "digital marketing"])
        )
        
        # Add complexity modifiers
        if complexity == "complex":
            description += f" with advanced {secrets.choice(['analytics', 'automation', 'integration', 'customization'])} capabilities"
        elif complexity == "moderate":
            description += f" offering {secrets.choice(['comprehensive', 'tailored', 'specialized'])} solutions"
        
        return description
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract key terms from business description."""
        # Simple keyword extraction - in practice, you might use NLP libraries
        words = re.findall(r'\b\w+\b', description.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))[:10]  # Limit to top 10 keywords
    
    def generate_domain_suggestions(self, business_desc: BusinessDescription, 
                                  num_suggestions: int = 3) -> List[DomainSuggestion]:
        """Generate domain name suggestions for a business description."""
        suggestions = []
        
        # Strategy 1: Keyword-based combinations
        if business_desc.keywords:
            keyword_suggestions = self._generate_keyword_domains(business_desc.keywords)
            suggestions.extend(keyword_suggestions[:num_suggestions//2])
        
        # Strategy 2: Industry-specific patterns
        industry_suggestions = self._generate_industry_domains(business_desc)
        suggestions.extend(industry_suggestions[:num_suggestions//2])
        
        # Strategy 3: Creative combinations
        creative_suggestions = self._generate_creative_domains(business_desc)
        suggestions.extend(creative_suggestions[:num_suggestions//2])
        
        # Ensure we have enough suggestions
        while len(suggestions) < num_suggestions:
            additional = self._generate_random_domains(business_desc)
            suggestions.extend(additional)
        
        # Remove duplicates and limit
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion.domain not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion.domain)
            if len(unique_suggestions) >= num_suggestions:
                break
        
        return unique_suggestions[:num_suggestions]
    
    def _generate_keyword_domains(self, keywords: List[str]) -> List[DomainSuggestion]:
        """Generate domains based on keyword combinations."""
        suggestions = []
        
        # Single keyword domains
        for keyword in keywords[:5]:
            if len(keyword) >= 3:
                domain = f"{keyword}.com"
                suggestions.append(DomainSuggestion(
                    domain=domain,
                    confidence=0.8,
                    reasoning=f"Direct keyword usage: {keyword}",
                    tld=".com"
                ))
        
        # Two-word combinations
        for i, word1 in enumerate(keywords[:3]):
            for word2 in keywords[i+1:4]:
                if len(word1) + len(word2) <= 15:
                    domain = f"{word1}{word2}.com"
                    suggestions.append(DomainSuggestion(
                        domain=domain,
                        confidence=0.7,
                        reasoning=f"Keyword combination: {word1} + {word2}",
                        tld=".com"
                    ))
        
        return suggestions
    
    def _generate_industry_domains(self, business_desc: BusinessDescription) -> List[DomainSuggestion]:
        """Generate industry-specific domain patterns."""
        suggestions = []
        
        industry_patterns = {
            "technology": ["tech", "digital", "smart", "ai", "data"],
            "healthcare": ["health", "care", "med", "wellness", "clinic"],
            "food_beverage": ["food", "eat", "dine", "kitchen", "cafe"],
            "consulting": ["consult", "advisory", "expert", "pro", "solutions"],
            "finance": ["finance", "money", "wealth", "invest", "bank"],
            "education": ["learn", "edu", "academy", "school", "study"]
        }
        
        patterns = industry_patterns.get(business_desc.industry, ["biz", "pro", "expert"])
        
        for pattern in patterns[:3]:
            # Combine with business type
            if business_desc.business_type != "startup":
                domain = f"{pattern}{business_desc.business_type.replace('_', '')}.com"
                suggestions.append(DomainSuggestion(
                    domain=domain,
                    confidence=0.75,
                    reasoning=f"Industry pattern + business type: {pattern} + {business_desc.business_type}",
                    tld=".com"
                ))
        
        return suggestions
    
    def _generate_creative_domains(self, business_desc: BusinessDescription) -> List[DomainSuggestion]:
        """Generate creative domain names."""
        suggestions = []
        
        # Use tone and target audience
        if business_desc.tone == "innovative":
            prefixes = ["next", "future", "innovate", "disrupt"]
        elif business_desc.tone == "luxury":
            prefixes = ["elite", "premium", "luxury", "exclusive"]
        elif business_desc.tone == "friendly":
            prefixes = ["friendly", "welcome", "hello", "nice"]
        else:
            prefixes = ["best", "top", "great", "excellent"]
        
        for prefix in prefixes[:2]:
            domain = f"{prefix}{business_desc.industry.replace('_', '')}.com"
            suggestions.append(DomainSuggestion(
                domain=domain,
                confidence=0.6,
                reasoning=f"Creative prefix + industry: {prefix} + {business_desc.industry}",
                tld=".com"
            ))
        
        return suggestions
    
    def _generate_random_domains(self, business_desc: BusinessDescription) -> List[DomainSuggestion]:
        """Generate random domain variations."""
        suggestions = []
        
        # Simple variations
        variations = [
            f"my{business_desc.industry.replace('_', '')}.com",
            f"get{business_desc.industry.replace('_', '')}.com",
            f"{business_desc.industry.replace('_', '')}pro.com"
        ]
        
        for domain in variations:
            suggestions.append(DomainSuggestion(
                domain=domain,
                confidence=0.5,
                reasoning="Random variation",
                tld=".com"
            ))
        
        return suggestions
    
    def create_training_dataset(self, num_samples: int = 10000, 
                              output_path: str = "data/synthetic/training_data.json") -> None:
        """Create the complete training dataset."""
        logger.info(f"Creating training dataset with {num_samples} samples")
        
        dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Generate business description
            business_desc = self.generate_business_description()
            
            # Generate domain suggestions
            domain_suggestions = self.generate_domain_suggestions(business_desc)
            
            # Create training examples
            for suggestion in domain_suggestions:
                example = {
                    "input": business_desc.description,
                    "output": suggestion.domain,
                    "metadata": {
                        "industry": business_desc.industry,
                        "business_type": business_desc.business_type,
                        "target_audience": business_desc.target_audience,
                        "tone": business_desc.tone,
                        "complexity": business_desc.complexity,
                        "keywords": business_desc.keywords,
                        "confidence": suggestion.confidence,
                        "reasoning": suggestion.reasoning,
                        "tld": suggestion.tld
                    }
                }
                dataset.append(example)
        
        # Save dataset
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total examples: {len(dataset)}")
        
        # Create summary statistics
        self._create_dataset_summary(dataset, output_path.replace('.json', '_summary.json'))
    
    def _create_dataset_summary(self, dataset: List[Dict], output_path: str) -> None:
        """Create summary statistics for the dataset."""
        df = pd.DataFrame(dataset)
        
        summary = {
            "total_examples": len(dataset),
            "unique_inputs": df['input'].nunique(),
            "unique_outputs": df['output'].nunique(),
            "industry_distribution": df['metadata'].apply(lambda x: x['industry']).value_counts().to_dict(),
            "business_type_distribution": df['metadata'].apply(lambda x: x['business_type']).value_counts().to_dict(),
            "tone_distribution": df['metadata'].apply(lambda x: x['tone']).value_counts().to_dict(),
            "complexity_distribution": df['metadata'].apply(lambda x: x['complexity']).value_counts().to_dict(),
            "tld_distribution": df['metadata'].apply(lambda x: x['tld']).value_counts().to_dict(),
            "average_confidence": df['metadata'].apply(lambda x: x['confidence']).mean(),
            "domain_length_stats": {
                "mean": df['output'].str.len().mean(),
                "std": df['output'].str.len().std(),
                "min": df['output'].str.len().min(),
                "max": df['output'].str.len().max()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {output_path}")


def main():
    """Main function to create the training dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create synthetic training dataset")
    parser.add_argument("--num_samples", type=int, default=10000, 
                       help="Number of samples to generate")
    parser.add_argument("--output_path", type=str, 
                       default="data/synthetic/training_data.json",
                       help="Output path for the dataset")
    parser.add_argument("--config_path", type=str, 
                       default="config/model_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset
    creator = SyntheticDatasetCreator(args.config_path)
    creator.create_training_dataset(args.num_samples, args.output_path)


if __name__ == "__main__":
    main() 
