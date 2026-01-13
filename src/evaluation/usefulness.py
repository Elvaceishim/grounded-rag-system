"""
Usefulness evaluation.

Measures whether answers are helpful, clear, and complete.
"""

from dataclasses import dataclass
import json
import re

from ..generation.prompts import USEFULNESS_PROMPT


@dataclass
class UsefulnessResult:
    """Result of usefulness evaluation."""
    
    score: float
    reasoning: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
        }


class UsefulnessEvaluator:
    """
    Evaluates answer usefulness using LLM-as-judge.
    
    Separate from faithfulness - an answer can be:
    - Faithful but useless (technically correct but doesn't help)
    - Useful but unfaithful (helpful but made up)
    """
    
    def __init__(self, generator):
        """
        Initialize the evaluator.
        
        Args:
            generator: Generator instance for LLM calls
        """
        self.generator = generator
    
    def evaluate(
        self,
        question: str,
        answer: str,
    ) -> UsefulnessResult:
        """
        Evaluate usefulness of an answer.
        
        Args:
            question: The original question
            answer: The generated answer
            
        Returns:
            UsefulnessResult with score and reasoning
        """
        prompt = USEFULNESS_PROMPT.format(
            question=question,
            answer=answer,
        )
        
        try:
            response = self.generator.call_llm_for_eval(prompt)
            result = self._parse_json_response(response)
            
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            
            return UsefulnessResult(
                score=round(score, 2),
                reasoning=reasoning,
            )
            
        except Exception as e:
            return UsefulnessResult(
                score=0.5,
                reasoning=f"Evaluation failed: {str(e)}",
            )
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            return json.loads(json_match.group())
        
        # Try parsing entire response
        return json.loads(response)
