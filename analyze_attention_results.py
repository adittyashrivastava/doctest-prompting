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
        print(f"   • Average Precision: {agg_metrics['avg_precision']:.3f}")
        print(f"   • Average Recall: {agg_metrics['avg_recall']:.3f}")  
        print(f"   • Average F1 Score: {agg_metrics['avg_f1']:.3f}")
        print(f"   • Average Top-K Accuracy: {agg_metrics['avg_top_k_accuracy']:.3f}")
        
        # Performance categorization
        f1_scores = [r['f1_score'] for r in self.summary_data['per_example_results']]
        excellent = sum(1 for f1 in f1_scores if f1 >= 0.8)
        good = sum(1 for f1 in f1_scores if 0.6 <= f1 < 0.8)
        moderate = sum(1 for f1 in f1_scores if 0.4 <= f1 < 0.6)
        poor = sum(1 for f1 in f1_scores if f1 < 0.4)
        total = len(f1_scores)
        
        print(f"\n📈 Performance Distribution:")
        print(f"   • Excellent (F1 ≥ 0.8): {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"   • Good (0.6 ≤ F1 < 0.8): {good}/{total} ({good/total*100:.1f}%)")
        print(f"   • Moderate (0.4 ≤ F1 < 0.6): {moderate}/{total} ({moderate/total*100:.1f}%)")
        print(f"   • Poor (F1 < 0.4): {poor}/{total} ({poor/total*100:.1f}%)")
        
        # Best and worst performers
        sorted_results = sorted(self.summary_data['per_example_results'], 
                              key=lambda x: x['f1_score'], reverse=True)
        
        print(f"\n🏆 Best Performers:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {result['example_id']}: F1={result['f1_score']:.3f}")
            
        print(f"\n⚠️  Worst Performers:")
        for i, result in enumerate(sorted_results[-3:]):
            print(f"   {i+1}. {result['example_id']}: F1={result['f1_score']:.3f}")
        
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
                avg_f1 = np.mean([r['f1_score'] for r in matching_results])
                avg_precision = np.mean([r['precision'] for r in matching_results])
                avg_recall = np.mean([r['recall'] for r in matching_results])
                
                print(f"   • {calc_type} ({len(matching_results)} examples):")
                print(f"     F1: {avg_f1:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
    
    def analyze_attention_patterns(self):
        """Analyze attention patterns and failure modes"""
        print(f"\n🧠 Attention Pattern Analysis:")
        
        # Analyze which facts are commonly missed
        missed_facts = {}
        correctly_identified = {}
        
        for result in self.detailed_data:
            ground_truth = result['ground_truth_facts']
            retrieved_texts = [fact.get('text', '') for fact in result['retrieved_facts']]
            
            for gt_fact in ground_truth:
                # Simple check if fact was retrieved (by keyword overlap)
                found = any(self.has_keyword_overlap(gt_fact, ret_text) for ret_text in retrieved_texts)
                
                if found:
                    correctly_identified[gt_fact] = correctly_identified.get(gt_fact, 0) + 1
                else:
                    missed_facts[gt_fact] = missed_facts.get(gt_fact, 0) + 1
        
        # Most commonly missed facts
        if missed_facts:
            sorted_missed = sorted(missed_facts.items(), key=lambda x: x[1], reverse=True)
            print(f"   🔴 Most commonly missed fact types:")
            for fact, count in sorted_missed[:5]:
                print(f"     • '{fact[:50]}...' (missed {count} times)")
        
        # Most commonly found facts  
        if correctly_identified:
            sorted_found = sorted(correctly_identified.items(), key=lambda x: x[1], reverse=True)
            print(f"   🟢 Most commonly identified fact types:")
            for fact, count in sorted_found[:5]:
                print(f"     • '{fact[:50]}...' (found {count} times)")
    
    def has_keyword_overlap(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """Check if two texts have significant keyword overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        return overlap / min(len(words1), len(words2)) >= threshold
    
    def generate_recommendations(self):
        """Generate recommendations for improving attention module"""
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("=" * 60)
        
        if not self.summary_data:
            return
            
        avg_f1 = self.summary_data['aggregate_metrics']['avg_f1']
        avg_precision = self.summary_data['aggregate_metrics']['avg_precision']
        avg_recall = self.summary_data['aggregate_metrics']['avg_recall']
        
        if avg_f1 >= 0.8:
            print("🎉 EXCELLENT: Your attention module is working very well!")
            print("   Consider:")
            print("   • Testing on more complex examples")
            print("   • Evaluating on larger datasets")
            print("   • Fine-tuning for specific medical domains")
            
        elif avg_f1 >= 0.6:
            print("✅ GOOD: Your attention module shows promising results.")
            print("   Areas for improvement:")
            if avg_precision < avg_recall:
                print("   • Precision is lower than recall - consider filtering irrelevant facts better")
                print("   • Adjust frequency_threshold or top_k parameters")
            else:
                print("   • Recall is lower than precision - consider expanding fact retrieval")
                print("   • Lower frequency_threshold to capture more facts")
            
        elif avg_f1 >= 0.4:
            print("🟡 MODERATE: Your attention module needs significant improvement.")
            print("   Key issues to address:")
            print("   • Consider adjusting layer_fraction (try 0.5 or 0.75)")
            print("   • Increase max_facts parameter")
            print("   • Review attention extraction methodology")
            print("   • Check if model architecture is suitable")
            
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
        
        if avg_recall < 0.5:
            print(f"   • Increase top_k from {current_k} to {int(current_k * 1.5) if isinstance(current_k, int) else 'higher'}")
            print("   • Lower frequency_threshold (try 0.9 or 0.85)")
            print("   • Increase max_facts parameter")
            
        if avg_precision < 0.5:
            print("   • Increase frequency_threshold (try 0.99)")
            print("   • Reduce max_facts to focus on most relevant")
            print("   • Consider post-processing to filter results")
    
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