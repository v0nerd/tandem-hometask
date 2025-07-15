"""
Safety Checker for Domain Name Generation

This module implements safety guardrails to prevent the generation of domains
for harmful or inappropriate content. It includes:
- Input classification and keyword filtering
- Rule-based rejection mechanisms
- Model-based content analysis
- Comprehensive logging of blocked requests
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """Result of safety check."""
    is_safe: bool
    risk_level: str  # "low", "medium", "high", "blocked"
    blocked_reasons: List[str]
    confidence: float
    metadata: Dict[str, Any]


class SafetyChecker:
    """Safety checker for domain name generation requests."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the safety checker."""
        self.config = self._load_config(config_path)
        self.blocked_keywords = self.config['safety']['blocked_keywords']
        self.blocked_domains = self.config['safety']['blocked_domains']
        self.min_confidence_threshold = self.config['safety']['min_confidence_threshold']
        
        # Load additional safety patterns
        self._load_safety_patterns()
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_safety_patterns(self):
        """Load additional safety patterns and rules."""
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(adult|porn|sex|nude|explicit)\b',
            r'\b(violence|hate|discrimination|racist)\b',
            r'\b(illegal|drugs|weapons|fraud)\b',
            r'\b(scam|phishing|malware|virus)\b',
            r'\b(terrorism|extremist|radical)\b'
        ]
        
        # High-risk business patterns
        self.high_risk_patterns = [
            r'\b(casino|gambling|betting|lottery)\b',
            r'\b(pyramid|mlm|multi.?level)\b',
            r'\b(counterfeit|fake|replica)\b',
            r'\b(hacking|cracking|warez)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_inappropriate = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.inappropriate_patterns]
        self.compiled_high_risk = [re.compile(pattern, re.IGNORECASE) 
                                  for pattern in self.high_risk_patterns]
    
    def _setup_logging(self):
        """Setup logging for safety events."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        safety_logger = logging.getLogger("safety_checker")
        safety_logger.setLevel(logging.INFO)
        
        # File handler for safety logs
        fh = logging.FileHandler(log_dir / "safety_events.log")
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        safety_logger.addHandler(fh)
        
        self.safety_logger = safety_logger
    
    def is_safe(self, business_description: str) -> bool:
        """Check if business description is safe for domain generation."""
        result = self.check_safety(business_description)
        return result.is_safe
    
    def check_safety(self, business_description: str) -> SafetyResult:
        """Comprehensive safety check for business description."""
        blocked_reasons = []
        risk_level = "low"
        confidence = 1.0
        
        # Check 1: Blocked keywords
        keyword_check = self._check_blocked_keywords(business_description)
        if not keyword_check['is_safe']:
            blocked_reasons.extend(keyword_check['reasons'])
            risk_level = "blocked"
            confidence = 0.0
        
        # Check 2: Inappropriate patterns
        pattern_check = self._check_inappropriate_patterns(business_description)
        if not pattern_check['is_safe']:
            blocked_reasons.extend(pattern_check['reasons'])
            if risk_level != "blocked":
                risk_level = "high"
            confidence = min(confidence, pattern_check['confidence'])
        
        # Check 3: High-risk business patterns
        high_risk_check = self._check_high_risk_patterns(business_description)
        if not high_risk_check['is_safe']:
            blocked_reasons.extend(high_risk_check['reasons'])
            if risk_level not in ["blocked", "high"]:
                risk_level = "medium"
            confidence = min(confidence, high_risk_check['confidence'])
        
        # Check 4: Content analysis
        content_check = self._analyze_content(business_description)
        if not content_check['is_safe']:
            blocked_reasons.extend(content_check['reasons'])
            if risk_level not in ["blocked", "high", "medium"]:
                risk_level = content_check['risk_level']
            confidence = min(confidence, content_check['confidence'])
        
        # Determine final safety
        is_safe = len(blocked_reasons) == 0 and confidence >= self.min_confidence_threshold
        
        # Log safety event
        self._log_safety_event(business_description, is_safe, risk_level, blocked_reasons)
        
        return SafetyResult(
            is_safe=is_safe,
            risk_level=risk_level,
            blocked_reasons=blocked_reasons,
            confidence=confidence,
            metadata={
                "timestamp": time.time(),
                "description_length": len(business_description),
                "checks_performed": ["keywords", "patterns", "high_risk", "content"]
            }
        )
    
    def _check_blocked_keywords(self, text: str) -> Dict[str, Any]:
        """Check for blocked keywords in text."""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.blocked_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            return {
                'is_safe': False,
                'reasons': [f"Contains blocked keyword: {keyword}" for keyword in found_keywords],
                'confidence': 0.0
            }
        
        return {
            'is_safe': True,
            'reasons': [],
            'confidence': 1.0
        }
    
    def _check_inappropriate_patterns(self, text: str) -> Dict[str, Any]:
        """Check for inappropriate content patterns."""
        found_patterns = []
        
        for pattern in self.compiled_inappropriate:
            if pattern.search(text):
                found_patterns.append(pattern.pattern)
        
        if found_patterns:
            return {
                'is_safe': False,
                'reasons': [f"Matches inappropriate pattern: {pattern}" for pattern in found_patterns],
                'confidence': 0.1
            }
        
        return {
            'is_safe': True,
            'reasons': [],
            'confidence': 1.0
        }
    
    def _check_high_risk_patterns(self, text: str) -> Dict[str, Any]:
        """Check for high-risk business patterns."""
        found_patterns = []
        
        for pattern in self.compiled_high_risk:
            if pattern.search(text):
                found_patterns.append(pattern.pattern)
        
        if found_patterns:
            return {
                'is_safe': False,
                'reasons': [f"Matches high-risk pattern: {pattern}" for pattern in found_patterns],
                'confidence': 0.3
            }
        
        return {
            'is_safe': True,
            'reasons': [],
            'confidence': 1.0
        }
    
    def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content for safety concerns."""
        concerns = []
        risk_level = "low"
        confidence = 1.0
        
        # Check for suspicious language patterns
        suspicious_indicators = [
            ("urgent", "high"),
            ("limited time", "medium"),
            ("guaranteed", "medium"),
            ("free money", "high"),
            ("get rich quick", "high"),
            ("no risk", "medium"),
            ("100% safe", "medium")
        ]
        
        text_lower = text.lower()
        for indicator, risk in suspicious_indicators:
            if indicator in text_lower:
                concerns.append(f"Suspicious language: {indicator}")
                if risk == "high":
                    risk_level = "high"
                    confidence = 0.2
                elif risk == "medium" and risk_level == "low":
                    risk_level = "medium"
                    confidence = 0.5
        
        # Check for excessive capitalization (potential spam)
        if text.isupper() and len(text) > 20:
            concerns.append("Excessive capitalization")
            if risk_level == "low":
                risk_level = "medium"
                confidence = 0.6
        
        # Check for repetitive patterns
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values()) if word_counts else 0
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                concerns.append("Excessive word repetition")
                if risk_level == "low":
                    risk_level = "medium"
                    confidence = 0.7
        
        return {
            'is_safe': len(concerns) == 0,
            'reasons': concerns,
            'risk_level': risk_level,
            'confidence': confidence
        }
    
    def _log_safety_event(self, description: str, is_safe: bool, risk_level: str, 
                         blocked_reasons: List[str]):
        """Log safety check event."""
        event = {
            "timestamp": time.time(),
            "is_safe": is_safe,
            "risk_level": risk_level,
            "blocked_reasons": blocked_reasons,
            "description_preview": description[:100] + "..." if len(description) > 100 else description
        }
        
        if not is_safe:
            self.safety_logger.warning(f"Unsafe content detected: {json.dumps(event)}")
        else:
            self.safety_logger.info(f"Safe content: {json.dumps(event)}")
    
    def filter_domain_suggestions(self, suggestions: List[Dict], 
                                business_description: str) -> List[Dict]:
        """Filter domain suggestions for safety."""
        filtered_suggestions = []
        
        for suggestion in suggestions:
            domain = suggestion.get('domain', '')
            
            # Check domain for blocked TLDs
            if any(tld in domain for tld in self.blocked_domains):
                logger.warning(f"Blocked domain TLD: {domain}")
                continue
            
            # Check domain for inappropriate content
            if self._is_domain_safe(domain, business_description):
                filtered_suggestions.append(suggestion)
            else:
                logger.warning(f"Unsafe domain filtered: {domain}")
        
        return filtered_suggestions
    
    def _is_domain_safe(self, domain: str, business_description: str) -> bool:
        """Check if a specific domain is safe."""
        domain_lower = domain.lower()
        
        # Check for blocked keywords in domain
        for keyword in self.blocked_keywords:
            if keyword.lower() in domain_lower:
                return False
        
        # Check for inappropriate patterns in domain
        for pattern in self.compiled_inappropriate:
            if pattern.search(domain):
                return False
        
        # Check for high-risk patterns in domain
        for pattern in self.compiled_high_risk:
            if pattern.search(domain):
                return False
        
        return True
    
    def get_safety_report(self, business_description: str) -> Dict[str, Any]:
        """Generate a comprehensive safety report."""
        result = self.check_safety(business_description)
        
        report = {
            "business_description": business_description,
            "safety_result": {
                "is_safe": result.is_safe,
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "blocked_reasons": result.blocked_reasons
            },
            "analysis": {
                "description_length": len(business_description),
                "word_count": len(business_description.split()),
                "has_suspicious_patterns": len(result.blocked_reasons) > 0,
                "risk_factors": self._identify_risk_factors(business_description)
            },
            "recommendations": self._generate_recommendations(result),
            "timestamp": time.time()
        }
        
        return report
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify specific risk factors in the text."""
        risk_factors = []
        text_lower = text.lower()
        
        # Check for various risk categories
        risk_categories = {
            "inappropriate_content": self.inappropriate_patterns,
            "high_risk_business": self.high_risk_patterns,
            "suspicious_language": [
                r'\b(urgent|limited time|guaranteed|free money)\b',
                r'\b(get rich quick|no risk|100% safe)\b'
            ]
        }
        
        for category, patterns in risk_categories.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    risk_factors.append(category)
                    break
        
        return list(set(risk_factors))
    
    def _generate_recommendations(self, result: SafetyResult) -> List[str]:
        """Generate recommendations based on safety check result."""
        recommendations = []
        
        if not result.is_safe:
            recommendations.append("Content blocked due to safety concerns")
            
            if "inappropriate_content" in result.metadata.get("risk_factors", []):
                recommendations.append("Avoid inappropriate or offensive content")
            
            if "high_risk_business" in result.metadata.get("risk_factors", []):
                recommendations.append("Consider the legal and ethical implications of the business")
            
            if "suspicious_language" in result.metadata.get("risk_factors", []):
                recommendations.append("Use clear, honest language without misleading claims")
        else:
            if result.risk_level == "medium":
                recommendations.append("Content is acceptable but consider reviewing for clarity")
            else:
                recommendations.append("Content appears safe for domain generation")
        
        return recommendations
    
    def update_safety_rules(self, new_blocked_keywords: List[str] = None,
                          new_blocked_domains: List[str] = None):
        """Update safety rules dynamically."""
        if new_blocked_keywords:
            self.blocked_keywords.extend(new_blocked_keywords)
            self.blocked_keywords = list(set(self.blocked_keywords))  # Remove duplicates
        
        if new_blocked_domains:
            self.blocked_domains.extend(new_blocked_domains)
            self.blocked_domains = list(set(self.blocked_domains))  # Remove duplicates
        
        logger.info(f"Updated safety rules: {len(self.blocked_keywords)} keywords, {len(self.blocked_domains)} domains")


def main():
    """Main function for testing the safety checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safety Checker")
    parser.add_argument("--description", type=str, required=True,
                       help="Business description to check")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="Path to config file")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed safety report")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize safety checker
    checker = SafetyChecker(args.config)
    
    # Check safety
    if args.report:
        report = checker.get_safety_report(args.description)
        print(json.dumps(report, indent=2))
    else:
        result = checker.check_safety(args.description)
        print(f"Safe: {result.is_safe}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Confidence: {result.confidence}")
        if result.blocked_reasons:
            print("Blocked Reasons:")
            for reason in result.blocked_reasons:
                print(f"  - {reason}")


if __name__ == "__main__":
    main() 