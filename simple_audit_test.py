#!/usr/bin/env python3
"""
Simple test to demonstrate the MedCalc audit system for one example.
"""

import sys
import audit
from auditors.audit_medcalc import AuditMedcalcFormula

# Sample example data
example = {
    "input": "Calculate the QTc (Bazett) for a patient with QT interval of 330 msec and heart rate of 75 bpm",
    "output": [
        "Calling analyze_input()...",
        "...analyze_input returned ('Calculate QTc using Bazett formula', ['QT interval', 'Heart Rate'], 'QTc_bazett')",
        "Calling get_data(['QT interval', 'Heart Rate'])...",
        "...get_data returned [('QT interval', 330, 'msec'), ('Heart Rate', 75, 'beats per minute')]",
        "Calling insert_variables('QTc = QT / sqrt(RR)', [('QT interval', 330, 'msec'), ('Heart Rate', 75, 'beats per minute')])...",
        "...insert_variables returned 'QTc = 330 / sqrt(60/75)'",
        "Calling solve_formula('QTc = 330 / sqrt(60/75)')...",
        "...solve_formula returned 'QTc = 330 / sqrt(0.8)'",
        "Calling solve_formula('QTc = 330 / sqrt(0.8)')...",
        "...solve_formula returned 'QTc = 330 / 0.894'",
        "Calling solve_formula('QTc = 330 / 0.894')...",
        "...solve_formula returned 'QTc = 368.951'",
        "Final answer: 368.951"
    ],
    "y_hat": "368.951",
    "y": "368.951",
    "is_correct": True
}

def main():
    print("=== MedCalc Audit System Demo ===")
    print(f"Input: {example['input']}")
    print(f"Expected Output: {example['y']}")
    print(f"Actual Output: {example['y_hat']}")
    print(f"Is Correct: {example['is_correct']}")
    print()

    # Create a trace from the example
    trace = audit.Trace(
        trace_lines=example['output'],
        is_correct=example['is_correct'],
        index=0
    )

    print(f"=== Trace Processing ===")
    print(f"Number of steps parsed: {len(trace.steps)}")
    print(f"Final answer extracted: {trace.final_answer}")
    print(f"Processing errors: {len(trace.errors)}")

    if trace.errors:
        print("Errors:")
        for error in trace.errors:
            print(f"  - {error}")
    print()

    # Run the audit
    print("=== Running Audit ===")
    auditor = AuditMedcalcFormula(trace)
    auditor.run_audits()

    print(f"Number of audits run: {len(auditor.audits)}")
    print("Audit Results:")
    for audit_result in auditor.audits:
        status = "✅ PASS" if audit_result['passed'] else "❌ FAIL"
        print(f"  {status}: {audit_result['msg']}")

    print()
    print("=== Summary ===")
    passed_audits = sum(1 for a in auditor.audits if a['passed'])
    total_audits = len(auditor.audits)
    print(f"Passed: {passed_audits}/{total_audits} audits")
    print(f"Overall result: {'✅ SUCCESS' if passed_audits == total_audits else '❌ ISSUES FOUND'}")

    # Show the DataFrame structure
    print()
    print("=== Trace Steps DataFrame ===")
    print(auditor.df)

if __name__ == "__main__":
    main()