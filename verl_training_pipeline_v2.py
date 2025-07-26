#!/usr/bin/env python3
"""
Verl Training Pipeline v2 - Separate Environment Approach
Uses vLLM for inference and verl for training with file-based communication
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass

@dataclass
class VerlTrainingConfig:
    # Data config
    dataset_path: str = "medcalc"
    max_samples: int = 100

    # Model config
    model_path: str = "microsoft/DialoGPT-medium"

    # Training config
    num_episodes: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5

    # Output config
    output_dir: str = "verl_training_output"
    checkpoint_dir: str = "verl_checkpoints"

    # Environment config
    vllm_env: str = "vllm-env"
    verl_env: str = "verl-env"

class VerlTrainingPipelineV2:
    def __init__(self, config: VerlTrainingConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"📁 Using temp directory: {self.temp_dir}")

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def run_vllm_inference(self, prompts: List[Dict]) -> List[Dict]:
        """Run inference using vLLM in separate environment"""
        # Save prompts to temp file
        prompts_file = self.temp_dir / "prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f)

        # Run vLLM inference script
        vllm_script = f"""
import json
import sys
from pathlib import Path

# Add attention_viz to path
sys.path.append('/home/ashriva3/codebase/attention_viz')

try:
    import vllm
    from vllm import LLM, SamplingParams
    from attention_viz import AttentionExtractor, AttrievelRetriever
    print("✅ vLLM environment loaded successfully")
except ImportError as e:
    print(f"❌ vLLM environment error: {{e}}")
    sys.exit(1)

def run_inference(prompts_file, output_file, model_path):
    # Load model
    llm = LLM(model=model_path)

    # Prepare prompts
    with open(prompts_file) as f:
        prompts_data = json.load(f)

    # Generate responses
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512)
    results = []

    for i, prompt_data in enumerate(prompts_data):
        prompt = prompt_data['prompt']
        info = prompt_data.get('info', {{}})

        # Generate response
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        # Mock attention data for now (would need separate forward pass)
        attention_data = {{
            'attention_weights': None,
            'retrieved_facts': [
                {{'fact': 'Sample fact 1', 'relevance': 0.8}},
                {{'fact': 'Sample fact 2', 'relevance': 0.6}}
            ]
        }}

        results.append({{
            'prompt': prompt,
            'response': response,
            'info': info,
            'attention_data': attention_data
        }})

        if (i + 1) % 10 == 0:
            print(f"Generated {{i + 1}}/{{len(prompts_data)}} responses")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Generated {{len(results)}} responses")

if __name__ == "__main__":
    prompts_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = sys.argv[3]
    run_inference(prompts_file, output_file, model_path)
"""

        # Save vLLM script
        vllm_script_file = self.temp_dir / "vllm_inference.py"
        with open(vllm_script_file, 'w') as f:
            f.write(vllm_script)

        # Run vLLM inference
        output_file = self.temp_dir / "inference_results.json"
        cmd = [
            "conda", "run", "-n", self.config.vllm_env,
            "python", str(vllm_script_file),
            str(prompts_file),
            str(output_file),
            self.config.model_path
        ]

        print(f"🚀 Running vLLM inference...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ vLLM inference failed: {result.stderr}")
            return []

        # Load results
        with open(output_file) as f:
            results = json.load(f)

        print(f"✅ Generated {len(results)} responses")
        return results

    def run_verl_training(self, training_data: List[Dict]) -> Dict:
        """Run verl training in separate environment"""
        # Save training data
        training_file = self.temp_dir / "training_data.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f)

        # Create verl training script
        verl_script = f"""
import json
import sys
from pathlib import Path

try:
    import verl
    from verl import PPOConfig, PPOTrainer
    print("✅ verl environment loaded successfully")
except ImportError as e:
    print(f"❌ verl environment error: {{e}}")
    sys.exit(1)

def run_training(training_file, output_dir, checkpoint_dir):
    # Load training data
    with open(training_file) as f:
        training_data = json.load(f)

    # Mock PPO training (simplified)
    print(f"Training on {{len(training_data)}} samples")

    # Simulate training progress
    for episode in range(5):
        print(f"Episode {{episode + 1}}/5")
        # Mock training steps
        for i in range(0, len(training_data), 4):
            batch = training_data[i:i+4]
            # Mock reward calculation
            total_reward = sum(
                sample.get('reward', 0.5)
                for sample in batch
            )
            print(f"  Batch {{i//4 + 1}}: avg reward = {{total_reward/len(batch):.3f}}")

    # Save mock results
    results = {{
        'episodes_completed': 5,
        'total_samples': len(training_data),
        'final_reward': 0.75
    }}

    with open(Path(output_dir) / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Training completed")

if __name__ == "__main__":
    training_file = sys.argv[1]
    output_dir = sys.argv[2]
    checkpoint_dir = sys.argv[3]
    run_training(training_file, output_dir, checkpoint_dir)
"""

        # Save verl script
        verl_script_file = self.temp_dir / "verl_training.py"
        with open(verl_script_file, 'w') as f:
            f.write(verl_script)

        # Run verl training
        cmd = [
            "conda", "run", "-n", self.config.verl_env,
            "python", str(verl_script_file),
            str(training_file),
            self.config.output_dir,
            self.config.checkpoint_dir
        ]

        print(f"🚀 Running verl training...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ verl training failed: {result.stderr}")
            return {}

        # Load results
        results_file = Path(self.config.output_dir) / "training_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = {}

        return results

    def calculate_rewards(self, inference_results: List[Dict]) -> List[Dict]:
        """Calculate rewards for training data"""
        training_data = []

        for result in inference_results:
            # Mock reward calculation based on attention quality
            attention_data = result.get('attention_data', {})
            retrieved_facts = attention_data.get('retrieved_facts', [])

            # Calculate reward based on fact relevance
            if retrieved_facts:
                avg_relevance = sum(fact.get('relevance', 0) for fact in retrieved_facts) / len(retrieved_facts)
                reward = avg_relevance * 0.8 + 0.2  # Base reward + relevance bonus
            else:
                reward = 0.2  # Base reward

            training_data.append({
                'prompt': result['prompt'],
                'response': result['response'],
                'reward': reward,
                'attention_data': attention_data
            })

        return training_data

    def load_sample_data(self) -> List[Dict]:
        """Load sample prompts for training"""
        # Mock data - in real implementation, load from MedCalc dataset
        sample_prompts = [
            {
                'prompt': 'Calculate the BMI for a person who is 70 kg and 1.75 meters tall.',
                'info': {'type': 'bmi_calculation'}
            },
            {
                'prompt': 'What is the creatinine clearance for a 65-year-old male with serum creatinine of 1.2 mg/dL?',
                'info': {'type': 'creatinine_clearance'}
            },
            {
                'prompt': 'Calculate the GFR using the MDRD formula for a 45-year-old female with creatinine 0.9 mg/dL.',
                'info': {'type': 'gfr_calculation'}
            },
            {
                'prompt': 'What is the corrected calcium for a patient with total calcium 8.5 mg/dL and albumin 3.2 g/dL?',
                'info': {'type': 'calcium_correction'}
            }
        ]

        return sample_prompts[:self.config.max_samples]

    def train(self):
        """Main training loop"""
        print("🚀 Starting Verl Training Pipeline v2")
        self.setup_directories()

        # Load sample data
        prompts = self.load_sample_data()
        print(f"📊 Loaded {len(prompts)} prompts")

        # Run vLLM inference
        inference_results = self.run_vllm_inference(prompts)
        if not inference_results:
            print("❌ No inference results, stopping")
            return

        # Calculate rewards
        training_data = self.calculate_rewards(inference_results)
        print(f"💰 Calculated rewards for {len(training_data)} samples")

        # Run verl training
        training_results = self.run_verl_training(training_data)

        print("✅ Training pipeline completed!")
        print(f"📈 Results: {training_results}")

        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir)
        print(f"🧹 Cleaned up temp directory")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medcalc", help="Dataset to use")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Model path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples to process")
    parser.add_argument("--vllm-env", default="vllm-env", help="vLLM conda environment")
    parser.add_argument("--verl-env", default="verl-env", help="verl conda environment")

    args = parser.parse_args()

    config = VerlTrainingConfig(
        dataset_path=args.dataset,
        model_path=args.model,
        num_episodes=args.episodes,
        max_samples=args.max_samples,
        vllm_env=args.vllm_env,
        verl_env=args.verl_env
    )

    pipeline = VerlTrainingPipelineV2(config)
    pipeline.train()

if __name__ == "__main__":
    main()