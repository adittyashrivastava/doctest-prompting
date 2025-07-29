#!/usr/bin/env python3
"""
Analysis script for attention module test results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import seaborn as sns

class AttentionResultsAnalyzer:
    """Analyze attention module test results"""
    
    def __init__(self, results_dir: str = "attention_test_results"):
        self.results_dir = Path(results_dir)
        self.summary_data = None
        self.detailed_data = None
        
    def load_results(self):
        """Load test results from files"""
        try:
            with open(self.results_dir / "summary.json", 'r') as f:
                self.summary_data = json.load(f)
            
            with open(self.results_dir / "detailed_results.json", 'r') as f:
                self.detailed_data = json.load(f)
                
            print("✅ Results loaded successfully")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Results files not found: {e}")
            return False
    
    def print_detailed_analysis(self):
        """Print detailed analysis of results"""
        if not self.summary_data or not self.detailed_data:
            print("❌ No results data loaded")
            return
            
        print("\n🔍 DETAILED ATTENTION MODULE ANALYSIS")
        print("=" * 60)
        
        # Overall metrics
        agg_metrics = self.summary_data['aggregate_metrics']
        print(f"📊 Overall Performance:")
        print(f"   • Average Top-K Containment Score: {agg_metrics['avg_top_k_containment']:.3f}")
        
        # Performance categorization
        containment_scores = [r['top_k_containment_score'] for r in self.summary_data['per_example_results']]
        excellent = sum(1 for score in containment_scores if score >= 0.8)
        good = sum(1 for score in containment_scores if 0.6 <= score < 0.8)
        moderate = sum(1 for score in containment_scores if 0.4 <= score < 0.6)
        poor = sum(1 for score in containment_scores if score < 0.4)
        total = len(containment_scores)
        
        print(f"\n📈 Performance Distribution:")
        print(f"   • Excellent (Containment ≥ 0.8): {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"   • Good (0.6 ≤ Containment < 0.8): {good}/{total} ({good/total*100:.1f}%)")
        print(f"   • Moderate (0.4 ≤ Containment < 0.6): {moderate}/{total} ({moderate/total*100:.1f}%)")
        print(f"   • Poor (Containment < 0.4): {poor}/{total} ({poor/total*100:.1f}%)")
        
        # Best and worst performers
        sorted_results = sorted(self.summary_data['per_example_results'], 
                              key=lambda x: x['top_k_containment_score'], reverse=True)
        
        print(f"\n🏆 Best Performers:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {result['example_id']}: Containment={result['top_k_containment_score']:.3f}")
            
        print(f"\n⚠️  Worst Performers:")
        for i, result in enumerate(sorted_results[-3:]):
            print(f"   {i+1}. {result['example_id']}: Containment={result['top_k_containment_score']:.3f}")
        
        # Detailed fact analysis
        print(f"\n🔍 Fact Retrieval Analysis:")
        total_retrieved = sum(len(r['retrieved_facts']) for r in self.detailed_data)
        total_ground_truth = sum(len(r['ground_truth_facts']) for r in self.detailed_data)
        
        print(f"   • Total facts retrieved: {total_retrieved}")
        print(f"   • Total ground truth facts: {total_ground_truth}")
        print(f"   • Average facts per example: {total_retrieved/len(self.detailed_data):.1f}")
        
        # Analysis by calculation type
        self.analyze_by_calculation_type()
        
    def analyze_by_calculation_type(self):
        """Analyze performance by type of medical calculation"""
        print(f"\n📋 Performance by Calculation Type:")
        
        # Group examples by type based on ID patterns
        calc_types = {
            'BMI/BSA': ['bmi_calc', 'body_surface_area'],
            'Renal Function': ['creatinine_clearance', 'gfr_mdrd', 'fractional_excretion'],
            'Electrolytes': ['corrected_calcium', 'anion_gap', 'osmolality', 'corrected_sodium'],
            'Cardiology': ['qtc_correction', 'cardiac_output'],
            'Fluid Balance': ['fluid_balance', 'maintenance_fluids'],
            'Risk Scores': ['wells_score', 'chads_vasc'],
            'Other': ['medication_dosing', 'insulin_sliding', 'acid_base', 'ideal_body_weight', 
                     'alveolar_arterial', 'parkland_formula']
        }
        
        for calc_type, patterns in calc_types.items():
            matching_results = []
            for result in self.summary_data['per_example_results']:
                if any(pattern in result['example_id'] for pattern in patterns):
                    matching_results.append(result)
            
            if matching_results:
                avg_containment = np.mean([r['top_k_containment_score'] for r in matching_results])
                
                print(f"   • {calc_type} ({len(matching_results)} examples):")
                print(f"     Containment Score: {avg_containment:.3f}")
    
    def analyze_attention_patterns(self):
        """Analyze attention patterns and failure modes"""
        print(f"\n🧠 Attention Pattern Analysis:")
        
        # Show examples of successful fact retrieval
        high_performers = [r for r in self.detailed_data if r['top_k_containment_score'] >= 0.7]
        low_performers = [r for r in self.detailed_data if r['top_k_containment_score'] < 0.3]
        
        print(f"   ✅ High-performing examples ({len(high_performers)}):")
        for result in high_performers[:3]:
            print(f"     • {result['example_id']}: Containment={result['top_k_containment_score']:.3f}")
            print(f"       Retrieved: {result['top_k_facts_text']}")
            if 'llm_response' in result:
                print(f"       🤖 LLM Response: {result['llm_response']}")
                print(f"       ✅ Expected: {result.get('expected_answer', 'N/A')}")
            
        print(f"   ❌ Low-performing examples ({len(low_performers)}):")
        for result in low_performers[:3]:
            print(f"     • {result['example_id']}: Containment={result['top_k_containment_score']:.3f}")
            print(f"       Expected Facts: {result['ground_truth_facts']}")
            print(f"       Retrieved: {result['top_k_facts_text']}")
            if 'llm_response' in result:
                print(f"       🤖 LLM Response: {result['llm_response']}")
                print(f"       ✅ Expected: {result.get('expected_answer', 'N/A')}")
    
    def generate_recommendations(self):
        """Generate recommendations for improving attention module"""
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("=" * 60)
        
        if not self.summary_data:
            return
            
        avg_containment = self.summary_data['aggregate_metrics']['avg_top_k_containment']
        
        if avg_containment >= 0.8:
            print("🎉 EXCELLENT: Your attention module is working very well!")
            print("   Consider:")
            print("   • Testing on more complex medical scenarios")
            print("   • Evaluating on larger datasets")
            print("   • Fine-tuning for specific medical domains")
            
        elif avg_containment >= 0.6:
            print("✅ GOOD: Your attention module shows promising results.")
            print("   Areas for improvement:")
            print("   • Fine-tune attention parameters to capture more relevant facts")
            print("   • Consider adjusting layer_fraction or top_k parameters")
            print("   • Test with different medical calculation types")
            
        elif avg_containment >= 0.4:
            print("🟡 MODERATE: Your attention module needs significant improvement.")
            print("   Key issues to address:")
            print("   • Consider adjusting layer_fraction (try 0.5 or 0.75)")
            print("   • Increase max_facts parameter")
            print("   • Review attention extraction methodology")
            print("   • Check if model architecture is suitable for medical text")
            
        else:
            print("❌ POOR: Your attention module needs major work.")
            print("   Critical issues:")
            print("   • Verify attention weights are being extracted correctly")
            print("   • Check tokenization alignment")
            print("   • Consider using a different model architecture")
            print("   • Review ATTRIEVAL configuration parameters")
        
        # Specific parameter recommendations
        print(f"\n🔧 Parameter Tuning Recommendations:")
        config = self.summary_data.get('test_config', {})
        current_k = config.get('k', 'unknown')
        
        if avg_containment < 0.5:
            print(f"   • Increase top_k from {current_k} to capture more facts")
            print("   • Lower frequency_threshold (try 0.9 or 0.85)")
            print("   • Increase max_facts parameter")
            print("   • Consider using more model layers (increase layer_fraction)")
        
        print(f"\n📊 Current Performance Summary:")
        print(f"   • Average Containment Score: {avg_containment:.3f}")
        print(f"   • This means {avg_containment*100:.1f}% of expected facts are being found in top {current_k} results")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        if not self.load_results():
            return False
            
        self.print_detailed_analysis()
        self.analyze_attention_patterns()
        self.generate_recommendations()
        
        return True

def main():
    """Main analysis function"""
    print("🔍 Attention Module Results Analysis")
    print("=" * 50)
    
    analyzer = AttentionResultsAnalyzer()
    
    if analyzer.run_full_analysis():
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Check the results in: {analyzer.results_dir}")
    else:
        print(f"❌ Analysis failed - make sure to run the attention test first")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 