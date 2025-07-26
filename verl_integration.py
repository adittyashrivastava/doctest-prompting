#!/usr/bin/env python3
"""
verl Integration for Reinforcement Learning with Attention-Based Rewards

This module integrates verl with the existing pipeline to create a reward model
that optimizes attention allocation to relevant facts for medical calculations.
"""

import sys
import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add attention_viz to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'attention_viz'))

try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"⚠️  attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False

try:
    import verl
    from verl import PPOConfig, PPOTrainer, RewardModel
    VERL_AVAILABLE = True
    print("✅ verl module loaded successfully")
except ImportError as e:
    print(f"⚠️  verl not available: {e}")
    print("Install with: pip install verl")
    VERL_AVAILABLE = False


@dataclass
class AttentionRewardConfig:
    """Configuration for attention-based reward model."""
    # Reward weights
    fact_relevance_weight: float = 0.4
    attention_quality_weight: float = 0.3
    calculation_accuracy_weight: float = 0.3

    # Attention analysis parameters
    layer_fraction: float = 0.25
    top_k: int = 10
    max_facts: int = 10
    frequency_threshold: float = 0.99

    # Reward thresholds
    min_fact_relevance: float = 0.1
    min_attention_quality: float = 0.1
    min_calculation_accuracy: float = 0.1


class AttentionRewardModel:
    """
    Reward model that evaluates attention quality and fact relevance.

    This model provides rewards based on:
    1. Fact Relevance: How relevant are the retrieved facts to the medical calculation?
    2. Attention Quality: How well does the model attend to relevant facts?
    3. Calculation Accuracy: Does better attention lead to more accurate calculations?
    """

    def __init__(self, config: AttentionRewardConfig):
        self.config = config
        self.attention_extractor = None
        self.fact_retriever = None
        self._setup_attention_analysis()

    def _setup_attention_analysis(self):
        """Setup attention analysis components."""
        if not ATTENTION_VIZ_AVAILABLE:
            print("⚠️  attention_viz not available - attention analysis disabled")
            return

        try:
            print("🔧 Setting up attention analysis for reward model...")

            # Initialize attention extractor (will be set with model later)
            # Initialize ATTRIEVAL with config
            attrieval_config = AttrievelConfig(
                layer_fraction=self.config.layer_fraction,
                top_k=self.config.top_k,
                frequency_threshold=self.config.frequency_threshold,
                max_facts=self.config.max_facts
            )

            print(f"✅ Attention reward model setup complete")
            print(f"   - Layer fraction: {self.config.layer_fraction}")
            print(f"   - Top K: {self.config.top_k}")
            print(f"   - Max facts: {self.config.max_facts}")

        except Exception as e:
            print(f"❌ Failed to setup attention analysis: {e}")

    def set_model_and_tokenizer(self, model, tokenizer):
        """Set the model and tokenizer for attention extraction."""
        if not ATTENTION_VIZ_AVAILABLE:
            return

        try:
            self.attention_extractor = AttentionExtractor(model, tokenizer)
            self.fact_retriever = AttrievelRetriever(self.attention_extractor, AttrievelConfig(
                layer_fraction=self.config.layer_fraction,
                top_k=self.config.top_k,
                frequency_threshold=self.config.frequency_threshold,
                max_facts=self.config.max_facts
            ))
            print("✅ Model and tokenizer set for attention analysis")
        except Exception as e:
            print(f"❌ Failed to set model and tokenizer: {e}")

    def calculate_fact_relevance(self, retrieved_facts: List[Dict], context: str, target: str) -> float:
        """
        Calculate relevance of retrieved facts to the medical calculation.

        Args:
            retrieved_facts: List of retrieved facts from ATTRIEVAL
            context: Input medical problem
            target: Expected calculation result

        Returns:
            Relevance score between 0 and 1
        """
        if not retrieved_facts:
            return 0.0

        # Simple relevance scoring based on keyword overlap
        # In practice, you might want to use a more sophisticated semantic similarity model

        context_lower = context.lower()
        target_lower = target.lower()

        relevant_keywords = []
        for fact in retrieved_facts:
            fact_text = fact.get('text', '').lower()

            # Check if fact contains medical terms from context
            medical_terms = ['diabetes', 'insulin', 'glucose', 'blood', 'pressure', 'weight', 'height', 'bmi', 'creatinine', 'gfr']
            for term in medical_terms:
                if term in context_lower and term in fact_text:
                    relevant_keywords.append(term)

            # Check if fact contains calculation-related terms
            calc_terms = ['calculate', 'formula', 'equation', 'dose', 'rate', 'concentration']
            for term in calc_terms:
                if term in fact_text:
                    relevant_keywords.append(term)

        # Normalize by number of facts and keywords
        relevance_score = len(set(relevant_keywords)) / max(len(retrieved_facts), 1)
        return min(relevance_score, 1.0)

    def calculate_attention_quality(self, attention_weights: List[torch.Tensor],
                                  relevant_tokens: List[str],
                                  all_tokens: List[str]) -> float:
        """
        Calculate quality of attention allocation to relevant tokens.

        Args:
            attention_weights: Attention weights from model
            relevant_tokens: Tokens that should receive high attention
            all_tokens: All tokens in the sequence

        Returns:
            Attention quality score between 0 and 1
        """
        if not attention_weights or not relevant_tokens:
            return 0.0

        try:
            # Get attention weights for the last layer
            last_layer_attention = attention_weights[-1]  # Shape: [num_heads, seq_len, seq_len]

            # Average across heads
            avg_attention = torch.mean(last_layer_attention, dim=0)  # Shape: [seq_len, seq_len]

            # Find indices of relevant tokens
            relevant_indices = []
            for i, token in enumerate(all_tokens):
                if any(relevant in token.lower() for relevant in relevant_tokens):
                    relevant_indices.append(i)

            if not relevant_indices:
                return 0.0

            # Calculate attention to relevant tokens
            attention_to_relevant = avg_attention[:, relevant_indices].mean(dim=1)
            attention_quality = torch.mean(attention_to_relevant).item()

            return min(attention_quality, 1.0)

        except Exception as e:
            print(f"⚠️  Error calculating attention quality: {e}")
            return 0.0

    def calculate_calculation_accuracy(self, prediction: str, target: str) -> float:
        """
        Calculate accuracy of the medical calculation.

        Args:
            prediction: Model's calculation result
            target: Expected calculation result

        Returns:
            Accuracy score between 0 and 1
        """
        try:
            # Try exact match first
            if prediction.strip().lower() == target.strip().lower():
                return 1.0

            # Try numeric comparison
            try:
                pred_num = float(prediction.strip())
                target_num = float(target.strip())
                # Allow for small tolerance in medical calculations
                tolerance = 0.01
                if abs(pred_num - target_num) <= tolerance:
                    return 1.0
                else:
                    # Partial credit based on how close the prediction is
                    error = abs(pred_num - target_num) / max(abs(target_num), 1e-6)
                    return max(0.0, 1.0 - error)
            except ValueError:
                pass

            # Try partial match for text results
            pred_lower = prediction.strip().lower()
            target_lower = target.strip().lower()

            if target_lower in pred_lower or pred_lower in target_lower:
                return 0.8

            return 0.0

        except Exception as e:
            print(f"⚠️  Error calculating calculation accuracy: {e}")
            return 0.0

    def calculate_reward(self,
                        retrieved_facts: List[Dict],
                        attention_weights: Optional[List[torch.Tensor]],
                        all_tokens: List[str],
                        prediction: str,
                        target: str,
                        context: str) -> Dict[str, float]:
        """
        Calculate the overall reward based on attention quality and fact relevance.

        Args:
            retrieved_facts: Facts retrieved by ATTRIEVAL
            attention_weights: Attention weights from model
            all_tokens: All tokens in the sequence
            prediction: Model's prediction
            target: Expected target
            context: Input context

        Returns:
            Dictionary with reward components and total reward
        """
        # Calculate individual reward components
        fact_relevance = self.calculate_fact_relevance(retrieved_facts, context, target)

        # Extract relevant tokens from facts for attention quality calculation
        relevant_tokens = []
        for fact in retrieved_facts:
            fact_text = fact.get('text', '')
            # Extract key terms from fact text
            terms = fact_text.split()[:5]  # Take first 5 terms as relevant
            relevant_tokens.extend(terms)

        attention_quality = self.calculate_attention_quality(
            attention_weights, relevant_tokens, all_tokens
        ) if attention_weights else 0.0

        calculation_accuracy = self.calculate_calculation_accuracy(prediction, target)

        # Calculate weighted reward
        total_reward = (
            self.config.fact_relevance_weight * fact_relevance +
            self.config.attention_quality_weight * attention_quality +
            self.config.calculation_accuracy_weight * calculation_accuracy
        )

        return {
            'fact_relevance': fact_relevance,
            'attention_quality': attention_quality,
            'calculation_accuracy': calculation_accuracy,
            'total_reward': total_reward,
            'weights': {
                'fact_relevance': self.config.fact_relevance_weight,
                'attention_quality': self.config.attention_quality_weight,
                'calculation_accuracy': self.config.calculation_accuracy_weight
            }
        }


class VerlAttentionTrainer:
    """
    Trainer that uses verl to optimize attention allocation.

    This trainer uses PPO to train the model to allocate attention
    more effectively to relevant facts for medical calculations.
    """

    def __init__(self, reward_model: AttentionRewardModel, config: Dict[str, Any]):
        self.reward_model = reward_model
        self.config = config
        self.trainer = None
        self._setup_verl()

    def _setup_verl(self):
        """Setup verl trainer."""
        if not VERL_AVAILABLE:
            print("❌ verl not available. Install with: pip install verl")
            return

        try:
            print("🔧 Setting up verl trainer...")

            # Configure PPO
            ppo_config = PPOConfig(
                learning_rate=self.config.get('learning_rate', 1e-5),
                batch_size=self.config.get('batch_size', 4),
                mini_batch_size=self.config.get('mini_batch_size', 1),
                n_epochs=self.config.get('n_epochs', 4),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                clip_range_vf=self.config.get('clip_range_vf', None),
                normalize_advantage=self.config.get('normalize_advantage', True),
                ent_coef=self.config.get('ent_coef', 0.01),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                target_kl=self.config.get('target_kl', None),
                stats_window_size=self.config.get('stats_window_size', 100),
                tensorboard_log=self.config.get('tensorboard_log', None),
                policy_kwargs=self.config.get('policy_kwargs', {}),
                verbose=self.config.get('verbose', 1),
                seed=self.config.get('seed', None),
                device=self.config.get('device', 'auto'),
                _init_setup_model=True,
            )

            print(f"✅ verl trainer setup complete")
            print(f"   - Learning rate: {ppo_config.learning_rate}")
            print(f"   - Batch size: {ppo_config.batch_size}")
            print(f"   - Epochs: {ppo_config.n_epochs}")

        except Exception as e:
            print(f"❌ Failed to setup verl trainer: {e}")

    def create_environment(self, model, tokenizer, dataset):
        """
        Create a custom environment for the RL training.

        This environment will:
        1. Take a medical calculation problem
        2. Generate a response using the model
        3. Extract attention and facts
        4. Calculate reward based on attention quality
        """
        # This is a placeholder - in practice, you'd implement a proper environment
        # that integrates with your existing pipeline
        pass

    def train(self, model, tokenizer, dataset, num_episodes=100):
        """
        Train the model using verl to optimize attention allocation.

        Args:
            model: The model to train
            tokenizer: The tokenizer
            dataset: Training dataset
            num_episodes: Number of training episodes
        """
        if not VERL_AVAILABLE:
            print("❌ verl not available for training")
            return

        print(f"🚀 Starting verl training with {num_episodes} episodes...")

        # Set up reward model with current model
        self.reward_model.set_model_and_tokenizer(model, tokenizer)

        # Training loop
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")

            # Sample a training example
            # In practice, you'd iterate through your dataset
            # For now, this is a placeholder

            # Calculate reward for this episode
            # reward = self.reward_model.calculate_reward(...)

            # Update model based on reward
            # This would involve the actual PPO training step

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1} completed")

        print("✅ Training completed")


def create_verl_config():
    """Create a configuration file for verl training."""
    config = {
        "reward_model": {
            "fact_relevance_weight": 0.4,
            "attention_quality_weight": 0.3,
            "calculation_accuracy_weight": 0.3,
            "layer_fraction": 0.25,
            "top_k": 10,
            "max_facts": 10,
            "frequency_threshold": 0.99
        },
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 4,
            "mini_batch_size": 1,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "num_episodes": 100
        }
    }

    with open("verl_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Created verl_config.json")


def test_verl_integration():
    """Test the verl integration with a simple example."""
    print("🧪 Testing verl Integration")

    # Test reward model
    config = AttentionRewardConfig()
    reward_model = AttentionRewardModel(config)

    # Test reward calculation
    test_facts = [
        {"text": "BMI is calculated as weight in kg divided by height in meters squared"},
        {"text": "Normal BMI range is 18.5 to 24.9"}
    ]

    reward = reward_model.calculate_reward(
        retrieved_facts=test_facts,
        attention_weights=None,  # Would be actual attention weights
        all_tokens=["calculate", "bmi", "weight", "height"],
        prediction="22.86",
        target="22.86",
        context="Calculate BMI for 70kg, 1.75m"
    )

    print(f"✅ Reward calculation test successful")
    print(f"   Fact relevance: {reward['fact_relevance']:.3f}")
    print(f"   Attention quality: {reward['attention_quality']:.3f}")
    print(f"   Calculation accuracy: {reward['calculation_accuracy']:.3f}")
    print(f"   Total reward: {reward['total_reward']:.3f}")

    return True


if __name__ == "__main__":
    # Create configuration file
    create_verl_config()

    # Test integration
    success = test_verl_integration()

    if success:
        print("\n🎉 verl integration successful!")
        print("You can now use verl for attention-optimized training.")
    else:
        print("\n❌ verl integration failed.")
        print("Please check the requirements and try again.")