#!/usr/bin/env python3
"""
Analysis script for General English attention module test results
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

class GeneralEnglishResultsAnalyzer:
    """Analyze general English attention module test results"""
    
    def __init__(self, results_dir: str = "general_english_attention_results"):
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
            
        print("\n🔍 DETAILED GENERAL ENGLISH ATTENTION ANALYSIS")
        print("=" * 70)
        
        # Overall metrics
        agg_metrics = self.summary_data['aggregate_metrics']
        print(f"📊 Overall Performance:")
        print(f"   • Average Top-K Containment Score: {agg_metrics['avg_top_k_containment']:.3f}")
        
        # Performance by domain
        print(f"\n📋 Performance by Domain:")
        domain_breakdown = self.summary_data.get('domain_breakdown', {})
        for domain, metrics in sorted(domain_breakdown.items()):
            print(f"   • {domain.title()} ({metrics['count']} examples):")
            print(f"     Containment Score: {metrics['avg_top_k_containment']:.3f}")
        
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
        
        # Best and worst performers by domain
        sorted_results = sorted(self.summary_data['per_example_results'], 
                              key=lambda x: x['top_k_containment_score'], reverse=True)
        
        print(f"\n🏆 Best Performers:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"   {i+1}. {result['example_id']} ({result['domain']}): Containment={result['top_k_containment_score']:.3f}")
            
        print(f"\n⚠️  Worst Performers:")
        for i, result in enumerate(sorted_results[-5:]):
            print(f"   {i+1}. {result['example_id']} ({result['domain']}): Containment={result['top_k_containment_score']:.3f}")
        
        # Domain-specific analysis
        self.analyze_domain_patterns()
        
    def analyze_domain_patterns(self):
        """Analyze patterns across different domains"""
        print(f"\n📊 Domain-Specific Analysis:")
        
        domain_results = {}
        for result in self.detailed_data:
            domain = result['domain']
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
        
        # Find domains where attention works well vs poorly
        domain_performance = {}
        for domain, results in domain_results.items():
            avg_containment = np.mean([r['top_k_containment_score'] for r in results])
            domain_performance[domain] = {
                'containment': avg_containment,
                'count': len(results)
            }
        
        # Sort by containment performance
        sorted_domains = sorted(domain_performance.items(), key=lambda x: x[1]['containment'], reverse=True)
        
        print(f"   🟢 Best Performing Domains:")
        for domain, perf in sorted_domains[:3]:
            print(f"     • {domain.title()}: Containment={perf['containment']:.3f}")
            
        print(f"   🔴 Challenging Domains:")
        for domain, perf in sorted_domains[-3:]:
            print(f"     • {domain.title()}: Containment={perf['containment']:.3f}")
    
    def analyze_attention_patterns(self):
        """Analyze what types of facts the attention module captures well"""
        print(f"\n🧠 Attention Pattern Analysis:")
        
        # Analyze high and low performers with LLM responses
        high_performers = [r for r in self.detailed_data if r['top_k_containment_score'] >= 0.7]
        low_performers = [r for r in self.detailed_data if r['top_k_containment_score'] < 0.3]
        
        print(f"   ✅ High-performing examples ({len(high_performers)}):")
        for result in high_performers[:2]:
            print(f"     • {result['example_id']} ({result['domain']}): Containment={result['top_k_containment_score']:.3f}")
            print(f"       Retrieved: {result['top_k_facts_text']}")
            if 'llm_response' in result:
                print(f"       🤖 LLM Response: {result['llm_response']}")
                print(f"       ✅ Expected: {result.get('expected_answer', 'N/A')}")
            
        print(f"   ❌ Low-performing examples ({len(low_performers)}):")
        for result in low_performers[:2]:
            print(f"     • {result['example_id']} ({result['domain']}): Containment={result['top_k_containment_score']:.3f}")
            print(f"       Expected Facts: {result['ground_truth_facts']}")
            print(f"       Retrieved: {result['top_k_facts_text']}")
            if 'llm_response' in result:
                print(f"       🤖 LLM Response: {result['llm_response']}")
                print(f"       ✅ Expected: {result.get('expected_answer', 'N/A')}")
        
        # Analyze containment score distribution
        avg_containment = np.mean([r['top_k_containment_score'] for r in self.detailed_data])
        
        print(f"\n   📊 Overall Analysis:")
        print(f"   • Average containment score: {avg_containment:.3f}")
        
        if avg_containment >= 0.7:
            print("   → Attention module is working very well at identifying relevant facts")
        elif avg_containment >= 0.5:
            print("   → Attention module captures most relevant facts but has room for improvement")
        elif avg_containment >= 0.3:
            print("   → Attention module struggles to identify relevant facts consistently")
        else:
            print("   → Attention module needs significant improvement in fact identification")
    
    def generate_recommendations(self):
        """Generate recommendations for improving attention on general English text"""
        print(f"\n💡 RECOMMENDATIONS FOR GENERAL ENGLISH TEXT")
        print("=" * 70)
        
        if not self.summary_data:
            return
            
        avg_containment = self.summary_data['aggregate_metrics']['avg_top_k_containment']
        
        # Overall assessment
        if avg_containment >= 0.7:
            print("🎉 EXCELLENT: Your attention module works very well on general English text!")
            print("   Recommendations:")
            print("   • Test on more complex reasoning tasks")
            print("   • Evaluate on longer documents")
            print("   • Try multi-hop reasoning questions")
            
        elif avg_containment >= 0.5:
            print("✅ GOOD: Your attention module shows solid performance on general English text.")
            print("   Areas for improvement:")
            print("   • Fine-tune attention parameters to capture more relevant facts")
            print("   • Consider adjusting layer_fraction to include more layers")
            print("   • Increase max_facts parameter for broader coverage")
            print("   • Consider domain-specific fine-tuning for weaker domains")
            
        elif avg_containment >= 0.3:
            print("🟡 MODERATE: Your attention module needs improvement for general English text.")
            print("   Key issues to address:")
            print("   • Review attention extraction methodology")
            print("   • Consider using different layer combinations (try layer_fraction=0.5)")
            print("   • Increase max_facts parameter for broader coverage")
            print("   • Experiment with different model architectures")
            
        else:
            print("❌ POOR: Your attention module struggles with general English text.")
            print("   Critical improvements needed:")
            print("   • Verify attention weights extraction is working correctly")
            print("   • Check tokenization alignment between model and ATTRIEVAL")
            print("   • Consider pre-training on reading comprehension tasks")
            print("   • Review model architecture compatibility")
        
        # Domain-specific recommendations
        if hasattr(self, 'summary_data') and 'domain_breakdown' in self.summary_data:
            domain_breakdown = self.summary_data['domain_breakdown']
            weak_domains = [domain for domain, metrics in domain_breakdown.items() 
                          if metrics['avg_top_k_containment'] < 0.4]
            strong_domains = [domain for domain, metrics in domain_breakdown.items() 
                            if metrics['avg_top_k_containment'] >= 0.7]
            
            if weak_domains:
                print(f"\n🎯 Domain-Specific Recommendations:")
                print(f"   Weak domains: {', '.join(weak_domains)}")
                print(f"   • Consider domain-specific vocabulary expansion")
                print(f"   • Add domain-specific training examples")
                print(f"   • Review domain-specific reasoning patterns")
                
            if strong_domains:
                print(f"   Strong domains: {', '.join(strong_domains)}")
                print(f"   • Analyze successful patterns and apply to other domains")
                print(f"   • Use these domains for positive examples in training")
    
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
    print("🔍 General English Attention Module Results Analysis")
    print("=" * 60)
    
    analyzer = GeneralEnglishResultsAnalyzer()
    
    if analyzer.run_full_analysis():
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Check the results in: {analyzer.results_dir}")
    else:
        print(f"❌ Analysis failed - make sure to run the general English test first")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 