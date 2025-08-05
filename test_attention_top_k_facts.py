#!/usr/bin/env python3
"""
Comprehensive Test Suite for Attention Module - Top K Facts Retrieval

This test verifies that the attention module can correctly identify the top K most
relevant facts from a given context when answering specific questions.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import attention_viz if available
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"❌ attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False

@dataclass
class TestExample:
    """A test example with context, question, and expected relevant facts"""
    id: str
    context: str
    question: str
    expected_answer: str
    ground_truth_facts: List[str]  # Facts that should be identified as relevant
    irrelevant_facts: List[str]    # Facts that should NOT be identified as relevant
    description: str

@dataclass
class AttentionTestResult:
    """Results from testing attention on a single example"""
    example_id: str
    retrieved_facts: List[Dict]
    ground_truth_facts: List[str]
    irrelevant_facts: List[str]
    top_k_containment_score: float  # Fraction of ground truth facts found in top K
    top_k_facts_text: List[str]  # Text of top K retrieved facts
    llm_response: str  # Generated response from the LLM
    expected_answer: str  # Expected answer for comparison
    attention_scores: Dict

class AttentionFactTestSuite:
    """Test suite for evaluating attention-based fact retrieval"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", k: int = 5):
        self.model_name = model_name
        self.k = k
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.test_examples = []
        self.results = []
        
    def setup_model(self):
        """Load model and tokenizer with GPU optimization, setup attention extractor and retriever"""
        if not ATTENTION_VIZ_AVAILABLE:
            raise RuntimeError("attention_viz module is not available. Cannot run attention tests.")

        print(f"🔧 Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine if this is a large model for special handling
        is_large_model = '7B' in self.model_name or '13B' in self.model_name or '70B' in self.model_name
        
        # Memory optimization strategy based on model size and available memory
        if torch.cuda.is_available():
            print(f"🔧 Using CUDA with memory optimizations for model {self.model_name}")
            
            # Check if bitsandbytes is available for quantization
            try:
                import bitsandbytes as bnb
                use_quantization = True
                print("✅ bitsandbytes available - using quantization for memory efficiency")
            except ImportError:
                use_quantization = False
                print("⚠️  bitsandbytes not available - falling back to float16")
            
            if use_quantization and is_large_model:
                # Use 8-bit quantization for large models (significant memory savings)
                print(f"🎯 Loading {self.model_name} with 8-bit quantization (75% memory reduction)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    # Configure for attention extraction
                    output_attentions=False,  # We'll enable this during inference
                    attn_implementation="eager"  # Force eager attention for better extraction
                )
            elif use_quantization:
                # For smaller models, use 4-bit quantization for even better memory efficiency
                print(f"🎯 Loading {self.model_name} with 4-bit quantization (better for smaller models)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    # Configure for attention extraction
                    output_attentions=False,  # We'll enable this during inference
                    attn_implementation="eager"  # Force eager attention for better extraction
                )
            else:
                # Fallback to float16 if quantization is not available
                print(f"🎯 Loading {self.model_name} with float16 (50% memory reduction)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    offload_buffers=True,  # Fix OOM issue for large models
                    # Configure for attention extraction
                    output_attentions=False,  # We'll enable this during inference
                    attn_implementation="eager"  # Force eager attention for better extraction
                )
        else:
            print(f"🔧 Using CPU + float32 for model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                offload_buffers=True,  # Fix OOM issue for large models
                # Configure for attention extraction
                output_attentions=False,  # We'll enable this during inference
                attn_implementation="eager"  # Force eager attention for better extraction
            )
        
        # Ensure model is in eval mode for consistent attention patterns
        self.model.eval()
        
        print(f'Model {self.model_name} loaded successfully for attention analysis')
        print(f'Model device: {next(self.model.parameters()).device}')
        print(f'Model dtype: {next(self.model.parameters()).dtype}')
        
        # Clear any cached memory after model loading (optional - can disable for performance)
        if torch.cuda.is_available() and os.environ.get('DISABLE_CUDA_CACHE_CLEAR', '').lower() != 'true':
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache after model loading")
        elif torch.cuda.is_available():
            print(f"⚡ Skipping CUDA cache clear (DISABLE_CUDA_CACHE_CLEAR=true)")
        
        # Ensure attention outputs are enabled and using eager implementation
        self.model.config.output_attentions = True
        self.model.config._attn_implementation = "eager"
        
        # Setup attention analysis
        extractor = AttentionExtractor(self.model, self.tokenizer)
        config = AttrievelConfig(
            layer_fraction=0.25,
            top_k=self.k,
            frequency_threshold=0.95,
            max_facts=self.k * 2 # Allow retrieving more facts than top_k for recall calculation
        )
        self.retriever = AttrievelRetriever(extractor, config)
        print("✅ Model and attention retriever setup complete")
        
    def create_test_dataset(self):
        """Create a comprehensive test dataset with known ground truth"""
        self.test_examples = [
            # Medical calculation examples
            TestExample(
                id="bmi_calc_1",
                context=(
                    "Patient Information:\n"
                    "- Name: John Smith\n"
                    "- Age: 45 years\n"
                    "- Weight: 80 kg\n"
                    "- Height: 1.75 m\n"
                    "- Blood pressure: 120/80 mmHg\n"
                    "- Heart rate: 72 bpm\n"
                    "- Temperature: 37°C\n"
                    "- Diabetic: No\n"
                    "- Smoker: No\n"
                    "BMI Formula: BMI = weight(kg) / height(m)²"
                ),
                question="Calculate the BMI for this patient.",
                expected_answer="26.12",
                ground_truth_facts=[
                    "Weight: 80 kg",
                    "Height: 1.75 m", 
                    "BMI = weight(kg) / height(m)²"
                ],
                irrelevant_facts=[
                    "Age: 45 years",
                    "Blood pressure: 120/80 mmHg",
                    "Heart rate: 72 bpm",
                    "Temperature: 37°C",
                    "Diabetic: No",
                    "Smoker: No"
                ],
                description="BMI calculation requiring weight, height, and formula"
            ),
            
            TestExample(
                id="creatinine_clearance_1",
                context=(
                    "Patient Data:\n"
                    "- Gender: Male\n"
                    "- Age: 65 years\n"
                    "- Weight: 70 kg\n"
                    "- Serum creatinine: 1.2 mg/dL\n"
                    "- Blood urea nitrogen: 25 mg/dL\n"
                    "- Albumin: 4.0 g/dL\n"
                    "- Hemoglobin: 13.5 g/dL\n"
                    "Cockcroft-Gault Formula:\n"
                    "CrCl = [(140 - age) × weight] / (72 × serum_creatinine)\n"
                    "For females, multiply by 0.85"
                ),
                question="Calculate the creatinine clearance using Cockcroft-Gault formula.",
                expected_answer="73.6 mL/min",
                ground_truth_facts=[
                    "Age: 65 years",
                    "Weight: 70 kg",
                    "Serum creatinine: 1.2 mg/dL",
                    "Gender: Male",
                    "CrCl = [(140 - age) × weight] / (72 × serum_creatinine)"
                ],
                irrelevant_facts=[
                    "Blood urea nitrogen: 25 mg/dL",
                    "Albumin: 4.0 g/dL",
                    "Hemoglobin: 13.5 g/dL",
                    "For females, multiply by 0.85"
                ],
                description="Creatinine clearance calculation requiring age, weight, creatinine, gender, and formula"
            ),
            
            TestExample(
                id="corrected_calcium_1",
                context=(
                    "Laboratory Results:\n"
                    "- Total calcium: 8.5 mg/dL\n"
                    "- Ionized calcium: 1.15 mmol/L\n"
                    "- Albumin: 3.2 g/dL\n"
                    "- Total protein: 7.0 g/dL\n"
                    "- Phosphorus: 4.0 mg/dL\n"
                    "- Magnesium: 2.0 mg/dL\n"
                    "- Vitamin D: 30 ng/mL\n"
                    "Corrected Calcium Formula:\n"
                    "Corrected Ca = Measured Ca + 0.8 × (4.0 - albumin)"
                ),
                question="Calculate the corrected calcium level.",
                expected_answer="9.14 mg/dL",
                ground_truth_facts=[
                    "Total calcium: 8.5 mg/dL",
                    "Albumin: 3.2 g/dL",
                    "Corrected Ca = Measured Ca + 0.8 × (4.0 - albumin)"
                ],
                irrelevant_facts=[
                    "Ionized calcium: 1.15 mmol/L",
                    "Total protein: 7.0 g/dL",
                    "Phosphorus: 4.0 mg/dL",
                    "Magnesium: 2.0 mg/dL",
                    "Vitamin D: 30 ng/mL"
                ],
                description="Corrected calcium calculation requiring total calcium, albumin, and correction formula"
            ),
            
            TestExample(
                id="anion_gap_1",
                context=(
                    "Electrolyte Panel:\n"
                    "- Sodium (Na): 140 mEq/L\n"
                    "- Potassium (K): 4.0 mEq/L\n"
                    "- Chloride (Cl): 102 mEq/L\n"
                    "- Bicarbonate (HCO3): 24 mEq/L\n"
                    "- Glucose: 95 mg/dL\n"
                    "- BUN: 18 mg/dL\n"
                    "- Creatinine: 1.0 mg/dL\n"
                    "Anion Gap Formula:\n"
                    "AG = (Na + K) - (Cl + HCO3)"
                ),
                question="Calculate the anion gap.",
                expected_answer="18 mEq/L",
                ground_truth_facts=[
                    "Sodium (Na): 140 mEq/L",
                    "Potassium (K): 4.0 mEq/L",
                    "Chloride (Cl): 102 mEq/L",
                    "Bicarbonate (HCO3): 24 mEq/L",
                    "AG = (Na + K) - (Cl + HCO3)"
                ],
                irrelevant_facts=[
                    "Glucose: 95 mg/dL",
                    "BUN: 18 mg/dL",
                    "Creatinine: 1.0 mg/dL"
                ],
                description="Anion gap calculation requiring Na, K, Cl, HCO3, and formula"
            ),
            
            TestExample(
                id="medication_dosing_1",
                context=(
                    "Patient Profile:\n"
                    "- Age: 8 years old\n"
                    "- Weight: 25 kg\n"
                    "- Height: 120 cm\n"
                    "- Diagnosis: Pneumonia\n"
                    "- Allergies: Penicillin\n"
                    "- Kidney function: Normal\n"
                    "- Liver function: Normal\n"
                    "Medication: Amoxicillin\n"
                    "Dosing: 45 mg/kg/day divided into 3 doses\n"
                    "Maximum daily dose: 3000 mg"
                ),
                question="Calculate the appropriate amoxicillin dose per administration.",
                expected_answer="375 mg per dose",
                ground_truth_facts=[
                    "Weight: 25 kg",
                    "Dosing: 45 mg/kg/day divided into 3 doses",
                    "Medication: Amoxicillin"
                ],
                irrelevant_facts=[
                    "Age: 8 years old",
                    "Height: 120 cm",
                    "Diagnosis: Pneumonia",
                    "Allergies: Penicillin",
                    "Kidney function: Normal",
                    "Liver function: Normal",
                    "Maximum daily dose: 3000 mg"
                ],
                description="Medication dosing calculation requiring weight, dosing guidelines, and division"
            ),
            
            # Add 15 more diverse examples
            TestExample(
                id="qtc_correction_1",
                context=(
                    "ECG Results:\n"
                    "- Heart rate: 80 bpm\n"
                    "- PR interval: 160 ms\n"
                    "- QRS duration: 90 ms\n"
                    "- QT interval: 440 ms\n"
                    "- QTc (machine): 495 ms\n"
                    "- Rhythm: Sinus rhythm\n"
                    "- Axis: Normal\n"
                    "Bazett's Formula: QTc = QT / √(RR interval in seconds)\n"
                    "RR interval = 60/heart rate"
                ),
                question="Calculate the corrected QT interval using Bazett's formula.",
                expected_answer="508 ms",
                ground_truth_facts=[
                    "Heart rate: 80 bpm",
                    "QT interval: 440 ms",
                    "QTc = QT / √(RR interval in seconds)",
                    "RR interval = 60/heart rate"
                ],
                irrelevant_facts=[
                    "PR interval: 160 ms",
                    "QRS duration: 90 ms",
                    "QTc (machine): 495 ms",
                    "Rhythm: Sinus rhythm",
                    "Axis: Normal"
                ],
                description="QTc calculation requiring heart rate, QT interval, and Bazett's formula"
            ),
            
            TestExample(
                id="fluid_balance_1",
                context=(
                    "24-hour Fluid Balance:\n"
                    "Intake:\n"
                    "- Oral fluids: 1500 mL\n"
                    "- IV fluids: 2000 mL\n"
                    "- Medications in fluid: 200 mL\n"
                    "- Food moisture content: 800 mL\n"
                    "Output:\n"
                    "- Urine: 1800 mL\n"
                    "- Wound drainage: 300 mL\n"
                    "- Nasogastric suction: 400 mL\n"
                    "- Insensible losses: 800 mL\n"
                    "Patient weight yesterday: 70 kg\n"
                    "Patient weight today: 71.2 kg"
                ),
                question="Calculate the net fluid balance for 24 hours.",
                expected_answer="+1200 mL",
                ground_truth_facts=[
                    "Oral fluids: 1500 mL",
                    "IV fluids: 2000 mL", 
                    "Medications in fluid: 200 mL",
                    "Food moisture content: 800 mL",
                    "Urine: 1800 mL",
                    "Wound drainage: 300 mL",
                    "Nasogastric suction: 400 mL",
                    "Insensible losses: 800 mL"
                ],
                irrelevant_facts=[
                    "Patient weight yesterday: 70 kg",
                    "Patient weight today: 71.2 kg"
                ],
                description="Fluid balance calculation requiring all intake and output measurements"
            ),
            
            TestExample(
                id="insulin_sliding_scale_1",
                context=(
                    "Patient Information:\n"
                    "- Type 2 Diabetes\n"
                    "- Current blood glucose: 280 mg/dL\n"
                    "- Weight: 80 kg\n"
                    "- Last insulin: 6 hours ago\n"
                    "- Current medications: Metformin 1000mg BID\n"
                    "Sliding Scale Protocol:\n"
                    "- BG 150-200: 2 units\n"
                    "- BG 201-250: 4 units\n"
                    "- BG 251-300: 6 units\n"
                    "- BG 301-350: 8 units\n"
                    "- BG >350: Call physician"
                ),
                question="Determine the appropriate insulin dose according to sliding scale.",
                expected_answer="6 units",
                ground_truth_facts=[
                    "Current blood glucose: 280 mg/dL",
                    "BG 251-300: 6 units"
                ],
                irrelevant_facts=[
                    "Type 2 Diabetes",
                    "Weight: 80 kg",
                    "Last insulin: 6 hours ago",
                    "Current medications: Metformin 1000mg BID",
                    "BG 150-200: 2 units",
                    "BG 201-250: 4 units",
                    "BG 301-350: 8 units",
                    "BG >350: Call physician"
                ],
                description="Insulin dosing requiring current glucose level and corresponding protocol"
            ),
            
            TestExample(
                id="gfr_mdrd_1",
                context=(
                    "Patient Demographics:\n"
                    "- Age: 55 years\n"
                    "- Gender: Female\n"
                    "- Race: African American\n"
                    "- Weight: 65 kg\n"
                    "- Height: 160 cm\n"
                    "Laboratory:\n"
                    "- Serum creatinine: 1.4 mg/dL\n"
                    "- BUN: 30 mg/dL\n"
                    "- Albumin: 3.8 g/dL\n"
                    "MDRD Formula:\n"
                    "GFR = 175 × (creatinine)^(-1.154) × (age)^(-0.203)\n"
                    "× 0.742 (if female) × 1.212 (if African American)"
                ),
                question="Calculate GFR using the MDRD formula.",
                expected_answer="48.2 mL/min/1.73m²",
                ground_truth_facts=[
                    "Age: 55 years",
                    "Gender: Female", 
                    "Race: African American",
                    "Serum creatinine: 1.4 mg/dL",
                    "GFR = 175 × (creatinine)^(-1.154) × (age)^(-0.203)",
                    "× 0.742 (if female) × 1.212 (if African American)"
                ],
                irrelevant_facts=[
                    "Weight: 65 kg",
                    "Height: 160 cm",
                    "BUN: 30 mg/dL",
                    "Albumin: 3.8 g/dL"
                ],
                description="MDRD GFR calculation requiring age, gender, race, creatinine, and formula"
            ),
            
            TestExample(
                id="acid_base_1",
                context=(
                    "Arterial Blood Gas Results:\n"
                    "- pH: 7.25\n"
                    "- PaCO2: 55 mmHg\n"
                    "- PaO2: 85 mmHg\n"
                    "- HCO3: 24 mEq/L\n"
                    "- Base excess: -2\n"
                    "- SaO2: 96%\n"
                    "- Lactate: 1.8 mmol/L\n"
                    "Normal Values:\n"
                    "- pH: 7.35-7.45\n"
                    "- PaCO2: 35-45 mmHg\n"
                    "- HCO3: 22-26 mEq/L\n"
                    "Winter's Formula: Expected PaCO2 = 1.5(HCO3) + 8 ± 2"
                ),
                question="Determine the primary acid-base disorder.",
                expected_answer="Respiratory acidosis",
                ground_truth_facts=[
                    "pH: 7.25",
                    "PaCO2: 55 mmHg",
                    "pH: 7.35-7.45",
                    "PaCO2: 35-45 mmHg"
                ],
                irrelevant_facts=[
                    "PaO2: 85 mmHg",
                    "HCO3: 24 mEq/L",
                    "Base excess: -2",
                    "SaO2: 96%",
                    "Lactate: 1.8 mmol/L",
                    "HCO3: 22-26 mEq/L",
                    "Winter's Formula: Expected PaCO2 = 1.5(HCO3) + 8 ± 2"
                ],
                description="Acid-base analysis requiring pH and PaCO2 compared to normal ranges"
            ),
            
            # Add 10 more examples to reach 20 total
            TestExample(
                id="osmolality_1",
                context=(
                    "Laboratory Values:\n"
                    "- Sodium: 145 mEq/L\n"
                    "- Glucose: 200 mg/dL\n"
                    "- BUN: 28 mg/dL\n"
                    "- Potassium: 4.2 mEq/L\n"
                    "- Chloride: 105 mEq/L\n"
                    "- CO2: 22 mEq/L\n"
                    "Calculated Osmolality Formula:\n"
                    "Osm = 2(Na) + (Glucose/18) + (BUN/2.8)"
                ),
                question="Calculate the serum osmolality.",
                expected_answer="311 mOsm/kg",
                ground_truth_facts=[
                    "Sodium: 145 mEq/L",
                    "Glucose: 200 mg/dL",
                    "BUN: 28 mg/dL",
                    "Osm = 2(Na) + (Glucose/18) + (BUN/2.8)"
                ],
                irrelevant_facts=[
                    "Potassium: 4.2 mEq/L",
                    "Chloride: 105 mEq/L",
                    "CO2: 22 mEq/L"
                ],
                description="Osmolality calculation requiring sodium, glucose, BUN, and formula"
            ),
            
            # Add more examples to reach 20 total
            TestExample(
                id="body_surface_area_1",
                context=(
                    "Patient Measurements:\n"
                    "- Weight: 70 kg\n"
                    "- Height: 175 cm\n"
                    "- Age: 40 years\n"
                    "- Gender: Male\n"
                    "- Blood pressure: 130/85 mmHg\n"
                    "BSA Formula (Mosteller): BSA = √[(height_cm × weight_kg)/3600]"
                ),
                question="Calculate the body surface area using Mosteller formula.",
                expected_answer="1.85 m²",
                ground_truth_facts=[
                    "Weight: 70 kg",
                    "Height: 175 cm",
                    "BSA = √[(height_cm × weight_kg)/3600]"
                ],
                irrelevant_facts=[
                    "Age: 40 years",
                    "Gender: Male",
                    "Blood pressure: 130/85 mmHg"
                ],
                description="BSA calculation requiring weight, height, and Mosteller formula"
            ),
            
            TestExample(
                id="corrected_sodium_1",
                context=(
                    "Laboratory Results:\n"
                    "- Measured sodium: 130 mEq/L\n"
                    "- Glucose: 400 mg/dL\n"
                    "- Potassium: 3.8 mEq/L\n"
                    "- Chloride: 95 mEq/L\n"
                    "- BUN: 22 mg/dL\n"
                    "- Creatinine: 1.1 mg/dL\n"
                    "Sodium Correction Formula:\n"
                    "Corrected Na = Measured Na + 0.016 × (glucose - 100)"
                ),
                question="Calculate the corrected sodium for hyperglycemia.",
                expected_answer="134.8 mEq/L",
                ground_truth_facts=[
                    "Measured sodium: 130 mEq/L",
                    "Glucose: 400 mg/dL",
                    "Corrected Na = Measured Na + 0.016 × (glucose - 100)"
                ],
                irrelevant_facts=[
                    "Potassium: 3.8 mEq/L",
                    "Chloride: 95 mEq/L",
                    "BUN: 22 mg/dL",
                    "Creatinine: 1.1 mg/dL"
                ],
                description="Corrected sodium calculation for hyperglycemia"
            ),
            
            TestExample(
                id="fractional_excretion_sodium_1",
                context=(
                    "Laboratory Data:\n"
                    "- Serum sodium: 140 mEq/L\n"
                    "- Urine sodium: 20 mEq/L\n"
                    "- Serum creatinine: 2.0 mg/dL\n"
                    "- Urine creatinine: 80 mg/dL\n"
                    "- BUN: 40 mg/dL\n"
                    "- Urine urea: 800 mg/dL\n"
                    "FENa Formula:\n"
                    "FENa = (Urine Na × Serum Cr) / (Serum Na × Urine Cr) × 100"
                ),
                question="Calculate the fractional excretion of sodium (FENa).",
                expected_answer="1.43%",
                ground_truth_facts=[
                    "Serum sodium: 140 mEq/L",
                    "Urine sodium: 20 mEq/L",
                    "Serum creatinine: 2.0 mg/dL",
                    "Urine creatinine: 80 mg/dL",
                    "FENa = (Urine Na × Serum Cr) / (Serum Na × Urine Cr) × 100"
                ],
                irrelevant_facts=[
                    "BUN: 40 mg/dL",
                    "Urine urea: 800 mg/dL"
                ],
                description="FENa calculation requiring sodium and creatinine values from serum and urine"
            ),
            
            TestExample(
                id="ideal_body_weight_1",
                context=(
                    "Patient Data:\n"
                    "- Gender: Female\n"
                    "- Height: 165 cm (5'5\")\n"
                    "- Current weight: 85 kg\n"
                    "- Age: 35 years\n"
                    "- BMI: 31.2\n"
                    "- Diagnosis: Obesity\n"
                    "IBW Formula (Robinson):\n"
                    "Female: IBW = 49 + 1.7 × (height_cm - 152.4)/2.54\n"
                    "Male: IBW = 52 + 1.9 × (height_cm - 152.4)/2.54"
                ),
                question="Calculate the ideal body weight using Robinson formula.",
                expected_answer="58.4 kg",
                ground_truth_facts=[
                    "Gender: Female",
                    "Height: 165 cm",
                    "Female: IBW = 49 + 1.7 × (height_cm - 152.4)/2.54"
                ],
                irrelevant_facts=[
                    "Current weight: 85 kg",
                    "Age: 35 years",
                    "BMI: 31.2",
                    "Diagnosis: Obesity",
                    "Male: IBW = 52 + 1.9 × (height_cm - 152.4)/2.54"
                ],
                description="Ideal body weight calculation requiring gender, height, and appropriate formula"
            ),
            
            TestExample(
                id="maintenance_fluids_1",
                context=(
                    "Pediatric Patient:\n"
                    "- Age: 5 years\n"
                    "- Weight: 18 kg\n"
                    "- Height: 110 cm\n"
                    "- No dehydration\n"
                    "- NPO for surgery\n"
                    "- Normal kidney function\n"
                    "Holiday-Segar Method:\n"
                    "- First 10 kg: 100 mL/kg/day\n"
                    "- Next 10 kg: 50 mL/kg/day\n"
                    "- Each kg >20: 20 mL/kg/day"
                ),
                question="Calculate daily maintenance fluid requirement.",
                expected_answer="1400 mL/day",
                ground_truth_facts=[
                    "Weight: 18 kg",
                    "First 10 kg: 100 mL/kg/day",
                    "Next 10 kg: 50 mL/kg/day"
                ],
                irrelevant_facts=[
                    "Age: 5 years",
                    "Height: 110 cm",
                    "No dehydration",
                    "NPO for surgery",
                    "Normal kidney function",
                    "Each kg >20: 20 mL/kg/day"
                ],
                description="Maintenance fluid calculation using Holiday-Segar method"
            ),
            
            TestExample(
                id="alveolar_arterial_gradient_1",
                context=(
                    "ABG and Clinical Data:\n"
                    "- FiO2: 0.21 (room air)\n"
                    "- Barometric pressure: 760 mmHg\n"
                    "- PaO2: 75 mmHg\n"
                    "- PaCO2: 40 mmHg\n"
                    "- pH: 7.40\n"
                    "- Temperature: 37°C\n"
                    "- RQ: 0.8\n"
                    "A-a Gradient Formula:\n"
                    "PAO2 = (FiO2 × (Patm - 47)) - (PaCO2/RQ)\n"
                    "A-a gradient = PAO2 - PaO2"
                ),
                question="Calculate the alveolar-arterial oxygen gradient.",
                expected_answer="25 mmHg",
                ground_truth_facts=[
                    "FiO2: 0.21",
                    "Barometric pressure: 760 mmHg",
                    "PaO2: 75 mmHg",
                    "PaCO2: 40 mmHg",
                    "RQ: 0.8",
                    "PAO2 = (FiO2 × (Patm - 47)) - (PaCO2/RQ)",
                    "A-a gradient = PAO2 - PaO2"
                ],
                irrelevant_facts=[
                    "pH: 7.40",
                    "Temperature: 37°C"
                ],
                description="A-a gradient calculation requiring multiple respiratory parameters"
            ),
            
            TestExample(
                id="cardiac_output_fick_1",
                context=(
                    "Cardiac Catheterization Data:\n"
                    "- Oxygen consumption: 250 mL/min\n"
                    "- Hemoglobin: 12 g/dL\n"
                    "- SaO2: 98%\n"
                    "- SvO2: 65%\n"
                    "- Heart rate: 75 bpm\n"
                    "- Blood pressure: 120/80 mmHg\n"
                    "Fick Equation:\n"
                    "CO = VO2 / (Hb × 1.36 × (SaO2 - SvO2))"
                ),
                question="Calculate cardiac output using Fick equation.",
                expected_answer="4.6 L/min",
                ground_truth_facts=[
                    "Oxygen consumption: 250 mL/min",
                    "Hemoglobin: 12 g/dL",
                    "SaO2: 98%",
                    "SvO2: 65%",
                    "CO = VO2 / (Hb × 1.36 × (SaO2 - SvO2))"
                ],
                irrelevant_facts=[
                    "Heart rate: 75 bpm",
                    "Blood pressure: 120/80 mmHg"
                ],
                description="Cardiac output calculation using Fick equation"
            ),
            
            TestExample(
                id="parkland_formula_1",
                context=(
                    "Burn Patient:\n"
                    "- Weight: 70 kg\n"
                    "- Total body surface area burned: 40%\n"
                    "- Age: 30 years\n"
                    "- Time since burn: 2 hours\n"
                    "- No inhalation injury\n"
                    "- Baseline vital signs stable\n"
                    "Parkland Formula:\n"
                    "Fluid = 4 mL × weight(kg) × %TBSA burned\n"
                    "Give 50% in first 8 hours, 50% in next 16 hours"
                ),
                question="Calculate total fluid requirement for first 24 hours.",
                expected_answer="11.2 L",
                ground_truth_facts=[
                    "Weight: 70 kg",
                    "Total body surface area burned: 40%",
                    "Fluid = 4 mL × weight(kg) × %TBSA burned"
                ],
                irrelevant_facts=[
                    "Age: 30 years",
                    "Time since burn: 2 hours",
                    "No inhalation injury",
                    "Baseline vital signs stable",
                    "Give 50% in first 8 hours, 50% in next 16 hours"
                ],
                description="Parkland formula for burn fluid resuscitation"
            ),
            
            TestExample(
                id="wells_score_pe_1",
                context=(
                    "Clinical Assessment:\n"
                    "- Clinical signs of DVT: Yes (3 points)\n"
                    "- PE most likely diagnosis: No (0 points)\n"
                    "- Heart rate >100: Yes (1.5 points)\n"
                    "- Immobilization >3 days: Yes (1.5 points)\n"
                    "- Previous PE/DVT: No (0 points)\n"
                    "- Hemoptysis: No (0 points)\n"
                    "- Cancer: Yes (1 point)\n"
                    "Wells Score Interpretation:\n"
                    "- Low risk: ≤4 points\n"
                    "- High risk: >4 points"
                ),
                question="Calculate Wells score for pulmonary embolism.",
                expected_answer="7 points (High risk)",
                ground_truth_facts=[
                    "Clinical signs of DVT: Yes (3 points)",
                    "Heart rate >100: Yes (1.5 points)",
                    "Immobilization >3 days: Yes (1.5 points)",
                    "Cancer: Yes (1 point)"
                ],
                irrelevant_facts=[
                    "PE most likely diagnosis: No (0 points)",
                    "Previous PE/DVT: No (0 points)",
                    "Hemoptysis: No (0 points)",
                    "Low risk: ≤4 points",
                    "High risk: >4 points"
                ],
                description="Wells score calculation for PE risk assessment"
            ),
            
            TestExample(
                id="chads_vasc_score_1",
                context=(
                    "Patient Profile:\n"
                    "- Age: 68 years (1 point for 65-74)\n"
                    "- Gender: Female (1 point)\n"
                    "- Congestive heart failure: Yes (1 point)\n"
                    "- Hypertension: Yes (1 point)\n"
                    "- Diabetes: No (0 points)\n"
                    "- Stroke/TIA history: No (0 points)\n"
                    "- Vascular disease: Yes (1 point)\n"
                    "- Atrial fibrillation: Yes\n"
                    "CHA2DS2-VASc Scoring:\n"
                    "- 0-1: Low risk\n"
                    "- 2: Moderate risk\n"
                    "- ≥3: High risk"
                ),
                question="Calculate CHA2DS2-VASc score for stroke risk.",
                expected_answer="5 points (High risk)",
                ground_truth_facts=[
                    "Age: 68 years (1 point for 65-74)",
                    "Gender: Female (1 point)",
                    "Congestive heart failure: Yes (1 point)",
                    "Hypertension: Yes (1 point)",
                    "Vascular disease: Yes (1 point)"
                ],
                irrelevant_facts=[
                    "Diabetes: No (0 points)",
                    "Stroke/TIA history: No (0 points)",
                    "Atrial fibrillation: Yes",
                    "0-1: Low risk",
                    "2: Moderate risk",
                    "≥3: High risk"
                ],
                description="CHA2DS2-VASc score calculation for stroke risk in atrial fibrillation"
            )
        ]
        
        print(f"📊 Created {len(self.test_examples)} test examples")
        
    def evaluate_example(self, example: TestExample) -> AttentionTestResult:
        """Evaluate attention retrieval on a single example"""
        try:
            # First: Generate LLM response BEFORE attention extraction to avoid interference
            print(f"   🤖 Generating LLM response first...")
            llm_response = self.generate_llm_response(example.context, example.question)
            
            # Second: Extract attention-based facts
            print(f"   🔍 Extracting attention-based facts...")
            retrieval_result = self.retriever.retrieve_facts(
                context=example.context,
                question=example.question, 
                cot_response=example.expected_answer,
                use_cross_evaluation=True
            )
            
            retrieved_facts = retrieval_result.get('retrieved_facts', [])
            
            # Calculate metrics
            top_k_containment_score = self.calculate_top_k_containment_score(
                retrieved_facts[:self.k], example.ground_truth_facts
            )
            
            top_k_facts_text = [fact.get('text', '') for fact in retrieved_facts[:self.k]]
            
            return AttentionTestResult(
                example_id=example.id,
                retrieved_facts=retrieved_facts,
                ground_truth_facts=example.ground_truth_facts,
                irrelevant_facts=example.irrelevant_facts,
                top_k_containment_score=top_k_containment_score,
                top_k_facts_text=top_k_facts_text,
                llm_response=llm_response,
                expected_answer=example.expected_answer,
                attention_scores=retrieval_result
            )
            
        except Exception as e:
            print(f"   ❌ Failed to evaluate example: {e}")
            
            return AttentionTestResult(
                example_id=example.id,
                retrieved_facts=[],
                ground_truth_facts=example.ground_truth_facts,
                irrelevant_facts=example.irrelevant_facts,
                top_k_containment_score=0.0,
                top_k_facts_text=[],
                llm_response="[Error: Could not generate response]",
                expected_answer=example.expected_answer,
                attention_scores={}
            )
    
    def calculate_top_k_containment_score(self, top_k_facts, ground_truth_facts):
        """Calculate how many ground truth facts are contained in top K retrieved facts"""
        if not ground_truth_facts:
            return 1.0  # If no ground truth facts expected, perfect score
            
        if not top_k_facts:
            return 0.0  # If no facts retrieved, zero score
            
        top_k_texts = [fact.get('text', '') for fact in top_k_facts]
        
        found_count = 0
        for gt_fact in ground_truth_facts:
            # Check if the ground truth fact is contained in any of the top K retrieved facts
            if any(self.fact_contained_in_text(gt_fact, retrieved_text) for retrieved_text in top_k_texts):
                found_count += 1
                
        return found_count / len(ground_truth_facts)
    
    def fact_contained_in_text(self, fact: str, text: str) -> bool:
        """Check if a fact is contained within a larger text"""
        # Convert to lowercase for case-insensitive matching
        fact_lower = fact.lower().strip()
        text_lower = text.lower().strip()
        
        # Direct substring containment
        if fact_lower in text_lower:
            return True
            
        # Check for key terms overlap (more robust matching)
        fact_words = set(fact_lower.split())
        text_words = set(text_lower.split())
        
        # If most key words from the fact are found in the text, consider it contained
        # Use a threshold of 70% overlap for key terms
        if len(fact_words) > 0:
            overlap = len(fact_words.intersection(text_words))
            overlap_ratio = overlap / len(fact_words)
            return overlap_ratio >= 0.7
            
        return False
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_llm_response(self, context: str, question: str) -> str:
        """Generate LLM response for the given context and question"""
        try:
            # Create the prompt for the LLM
            prompt = f"You are an expert at processing medical data. You are given a context with a list of patient parameters and a question. You need to answer the question based on the context. Keep your answer concise and to the point. Do not exceed one sentence and do not exceed 15 words. Folow the scoring specification in the context carefully. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Debug: Check prompt length and content
            print(f"   📏 Prompt length: {len(prompt)} chars")
            if len(prompt) > 4000:  # Increased from 2000
                print(f"   ⚠️  Very long prompt detected, truncating...")
                # Truncate context if too long - but more generous
                max_context_len = 3500 - len(question) - 50  # More generous limit
                if len(context) > max_context_len:
                    context = context[:max_context_len] + "..."
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    print(f"   📏 Truncated prompt length: {len(prompt)} chars")
            
            # Tokenize the prompt
            print(f"   🔤 Tokenizing prompt...")
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,  # Increased from 512 for longer contexts
                padding=False,
                add_special_tokens=True
            )
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Validate input shapes
            if inputs['input_ids'].shape[1] == 0:
                return "[Error: Empty input after tokenization]"
            
            print(f"   📊 Token stats: shape={inputs['input_ids'].shape}")
            print(f"   🔧 Input device: {inputs['input_ids'].device}")
            print(f"   🔧 Model device: {device}")
            
            print(f"   🤖 Generating response...")
            # Generate response with optimized settings for GPU/CPU
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Increased to 300 for detailed responses
                    do_sample=False,  # Use greedy decoding for stability
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                print(f"   ✅ Generation completed successfully")
            
            # Validate output before decoding
            if outputs.shape[1] <= inputs['input_ids'].shape[1]:
                return "[Error: No new tokens generated]"
            
            # Decode the response (only the newly generated part)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            print(f"   📝 Generated: {generated_text[:100]}...")
            return generated_text if generated_text else "[Error: Empty generation]"
            
        except Exception as e:
            print(f"   ❌ Failed to generate LLM response: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def run_all_tests(self, num_examples=None):
        """Run tests on specified number of examples"""
        print("🚀 Starting comprehensive attention module testing")
        print("=" * 60)
        
        if not ATTENTION_VIZ_AVAILABLE:
            raise RuntimeError("attention_viz not available")
            
        # Setup
        self.setup_model()
        self.create_test_dataset()
        
        # Run tests
        self.results = []

        for example in self.test_examples[:num_examples]:
            print(f"   🔍 Running test {example.id}...")
            result = self.evaluate_example(example)
            self.results.append(result)
            
        # Generate summary
        self.generate_summary()
        self.save_results()
        
    def generate_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 ATTENTION MODULE TEST SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to analyze")
            return
            
        # Calculate aggregate metrics
        avg_top_k_containment = np.mean([r.top_k_containment_score for r in self.results])
        
        print(f"📈 Overall Performance (K={self.k}):")
        print(f"   Average Top-{self.k} Containment Score: {avg_top_k_containment:.3f}")
        
        # Per-example breakdown
        print(f"\n📋 Per-Example Results:")
        for result in self.results:
            print(f"   {result.example_id}:")
            print(f"     Top-{self.k} Containment Score: {result.top_k_containment_score:.3f}")
            print(f"     Top-{self.k} Facts: {', '.join(result.top_k_facts_text)}")
            
        # Success analysis
        high_performers = [r for r in self.results if r.top_k_containment_score >= 0.7]
        low_performers = [r for r in self.results if r.top_k_containment_score < 0.3]
        
        print(f"\n✅ High Performers (Containment ≥ 0.7): {len(high_performers)}/{len(self.results)}")
        for r in high_performers:
            print(f"   {r.example_id}: Containment={r.top_k_containment_score:.3f}")
            
        print(f"\n❌ Low Performers (Containment < 0.3): {len(low_performers)}/{len(self.results)}")
        for r in low_performers:
            print(f"   {r.example_id}: Containment={r.top_k_containment_score:.3f}")
            
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if avg_top_k_containment >= 0.7:
            print("   ✅ EXCELLENT: Attention module working very well")
        elif avg_top_k_containment >= 0.5:
            print("   🟡 GOOD: Attention module working reasonably well")
        elif avg_top_k_containment >= 0.3:
            print("   🟠 MODERATE: Attention module needs improvement")
        else:
            print("   ❌ POOR: Attention module needs significant work")
            
    def save_results(self):
        """Save detailed results to files"""
        output_dir = Path("attention_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_data = []
        for result in self.results:
            results_data.append({
                'example_id': result.example_id,
                'top_k_containment_score': result.top_k_containment_score,
                'top_k_facts_text': result.top_k_facts_text,
                'llm_response': result.llm_response,
                'expected_answer': result.expected_answer,
                'retrieved_facts': result.retrieved_facts,
                'ground_truth_facts': result.ground_truth_facts,
                'irrelevant_facts': result.irrelevant_facts
            })
            
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
            
        # Save summary statistics
        summary = {
            'test_config': {
                'model_name': self.model_name,
                'k': self.k,
                'num_examples': len(self.test_examples)
            },
            'aggregate_metrics': {
                'avg_top_k_containment': float(np.mean([r.top_k_containment_score for r in self.results])),
            },
            'per_example_results': [
                {
                    'example_id': r.example_id,
                    'top_k_containment_score': r.top_k_containment_score,
                    'top_k_facts_text': r.top_k_facts_text
                }
                for r in self.results
            ]
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   - Detailed results: {output_dir}/detailed_results.json")
        print(f"   - Summary: {output_dir}/summary.json")

    def cleanup(self):
        """Clean up memory after testing"""
        try:
            print("🧹 Cleaning up memory...")
            
            # Clear references to model and tokenizer
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            if hasattr(self, 'retriever') and self.retriever is not None:
                del self.retriever
                self.retriever = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("✅ Memory cleanup complete")
                
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            print("   Continuing anyway...")

def main():
    """Main function to run the attention module test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test attention module for top-K fact retrieval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to test")
    parser.add_argument("--k", type=int, default=5, help="Number of top facts to retrieve")
    parser.add_argument("--examples", type=int, default=20, help="Number of test examples to run")
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = AttentionFactTestSuite(model_name=args.model, k=args.k)
    
    try:
        test_suite.run_all_tests(num_examples=args.examples) #pass num_examples to run_all_tests
        print("\n🎉 Attention module testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 