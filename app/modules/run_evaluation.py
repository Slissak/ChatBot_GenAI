#!/usr/bin/env python3
"""
Simple script to run conversation evaluation with different options

Usage examples:
    python run_evaluation.py                           # Run full evaluation
    python run_evaluation.py --conversations 5         # Evaluate first 5 conversations only
    python run_evaluation.py --report custom_report.md # Custom report file name
"""

import sys
import argparse
from conversation_evaluator import ConversationEvaluator

def main():
    parser = argparse.ArgumentParser(description='Run conversation evaluation')
    parser.add_argument('--json-file', default='app/sms_conversations.json', 
                       help='Path to conversation JSON file')
    parser.add_argument('--report', default='evaluation_report.md',
                       help='Output report file name')
    parser.add_argument('--conversations', type=int, default=None,
                       help='Number of conversations to evaluate (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    print("🎯 Conversation Evaluation Tool")
    print("=" * 40)
    print(f"JSON File: {args.json_file}")
    print(f"Report File: {args.report}")
    print(f"Conversations: {'All' if args.conversations is None else args.conversations}")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(args.json_file)
    
    # Load conversations
    if not evaluator.load_conversations():
        print("❌ Failed to load conversations")
        return 1
    
    # Limit conversations if specified
    if args.conversations is not None:
        evaluator.conversations = evaluator.conversations[:args.conversations]
        print(f"📝 Limited to first {len(evaluator.conversations)} conversations")
    
    # Run evaluation
    print("🚀 Starting evaluation...")
    metrics = evaluator.evaluate_all_conversations()
    
    if metrics is None:
        print("❌ Evaluation failed")
        return 1
    
    # Generate report
    report = evaluator.generate_report(metrics, args.report)
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"📈 Total Conversations: {metrics.total_conversations}")
    print(f"🎯 Overall Accuracy: {metrics.total_accuracy:.1%}")
    print(f"🔚 End Prediction Accuracy: {metrics.end_prediction_accuracy:.1%}")
    print(f"📊 Confusion Matrix:")
    print(f"   True Positives:  {metrics.confusion_matrix['TP']}")
    print(f"   True Negatives:  {metrics.confusion_matrix['TN']}")
    print(f"   False Positives: {metrics.confusion_matrix['FP']}")
    print(f"   False Negatives: {metrics.confusion_matrix['FN']}")
    print(f"📄 Full report saved to: {args.report}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 