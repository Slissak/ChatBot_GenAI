# Conversation Evaluation System

## Overview

This module provides comprehensive evaluation capabilities for your conversation assistant application. It reads JSON conversation files, simulates conversations with your app, and evaluates accuracy using multiple metrics.

## Features

### Core Functionality
- ✅ **JSON Conversation Parsing**: Reads structured conversation data
- ✅ **Automated Testing**: Simulates conversations with your app
- ✅ **Accuracy Evaluation**: Calculates response appropriateness 
- ✅ **End Prediction Analysis**: Tests conversation ending decisions
- ✅ **Agent Usage Tracking**: Monitors which agents are used
- ✅ **Confusion Matrix**: Detailed accuracy breakdown
- ✅ **Comprehensive Reporting**: Detailed markdown reports

### Evaluation Metrics

#### 1. **Response Accuracy**
- Measures appropriateness of supervisor responses
- Uses heuristic analysis of response-to-input matching
- Considers response length, content relevance, and context

#### 2. **End Prediction Accuracy** 
- Tests conversation ending detection
- Compares expected vs actual END/NOT_END decisions
- Provides confusion matrix with TP, TN, FP, FN

#### 3. **Agent Usage Analysis**
- Tracks info_agent, sched_agent, exit_agent usage
- Identifies which agents respond to what types of queries
- Helps optimize agent delegation

## File Structure

```
app/modules/
├── conversation_evaluator.py  # Main evaluation engine
├── run_evaluation.py         # Simple CLI runner
└── README_EVALUATION.md      # This documentation
```

## Usage Examples

### Basic Usage

```python
from modules.conversation_evaluator import ConversationEvaluator

# Initialize evaluator
evaluator = ConversationEvaluator("sms_conversations.json")

# Load and evaluate all conversations
evaluator.load_conversations()
metrics = evaluator.evaluate_all_conversations()

# Generate report
report = evaluator.generate_report(metrics, "my_report.md")
print(f"Accuracy: {metrics.total_accuracy:.1%}")
```

### Command Line Usage

```bash
# Evaluate all conversations
python modules/run_evaluation.py

# Evaluate first 3 conversations only  
python modules/run_evaluation.py --conversations 3

# Custom report file
python modules/run_evaluation.py --report my_custom_report.md

# Custom JSON file
python modules/run_evaluation.py --json-file my_conversations.json
```

### Advanced Usage

```python
# Evaluate with custom app instance
from main import SupervisorAgent

supervisor = SupervisorAgent()
evaluator = ConversationEvaluator("conversations.json", app_instance=supervisor)

# Run evaluation and access detailed results
metrics = evaluator.evaluate_all_conversations()

# Access individual conversation results
for result in metrics.conversation_results:
    print(f"Conversation {result.conversation_id}: {result.accuracy:.1%}")
    print(f"Agents used: {result.agent_usage}")
    print(f"Errors: {result.errors}")
```

## Expected JSON Format

Your conversation JSON should follow this structure:

```json
[
  {
    "conversation_id": 1,
    "candidate_phone": "+1-555-0201",
    "recruiter_phone": "+1-555-0000", 
    "start_time_utc": "2024-04-03T15:12:00Z",
    "turns": [
      {
        "turn_id": 1,
        "speaker": "recruiter",
        "timestamp_utc": "2024-04-03T15:12:00Z",
        "text": "Thanks for applying to our Python Developer opening.",
        "label": "continue"
      },
      {
        "turn_id": 2,
        "speaker": "candidate", 
        "timestamp_utc": "2024-04-03T15:13:19Z",
        "text": "I've been using Python professionally for five years.",
        "label": null
      }
    ]
  }
]
```

### Required Fields
- `conversation_id`: Unique identifier
- `turns`: Array of conversation turns
- `turn_id`: Turn sequence number
- `speaker`: "candidate" or "recruiter"
- `text`: The message content
- `label`: "continue", "schedule", "end", or null

## Understanding the Results

### Accuracy Metrics

```
Overall Performance Summary
- Total Conversations Evaluated: 15
- Average Response Accuracy: 78.3%
- End Prediction Accuracy: 86.7%
```

### Confusion Matrix
```
                 Predicted  
              NOT_END   END
Actual NOT_END   8       1
       END       1       5
```

**Interpretation:**
- **True Negatives (8)**: Correctly predicted conversation continues
- **True Positives (5)**: Correctly predicted conversation ends  
- **False Positives (1)**: Incorrectly predicted end (premature ending)
- **False Negatives (1)**: Missed conversation ending

### Agent Usage Statistics
```
- Info Agent Usage: 12 times
- Scheduling Agent Usage: 18 times  
- Exit Agent Usage: 15 times
```

## Evaluation Criteria

### Response Appropriateness Heuristics

1. **Scheduling Queries**: If candidate mentions scheduling → response should contain scheduling keywords
2. **Job Information**: If candidate asks about job → response should be informative (>50 chars)
3. **Uninterested Candidates**: If candidate shows disinterest → response should be understanding
4. **General**: Response should be substantial (>20 chars) and error-free

### End Prediction Logic

- **Expected END**: Last recruiter turn has `"label": "end"`
- **Actual END**: App responds with "[END]" or exit agent decides to end
- **Accuracy**: Percentage of correct END/NOT_END predictions

## Customization

### Custom Evaluation Criteria

You can modify `_is_turn_correct()` in `conversation_evaluator.py` to implement your own response evaluation logic:

```python
def _is_turn_correct(self, turn: Dict) -> bool:
    """Custom evaluation logic"""
    response = turn['supervisor_response'].lower() 
    candidate_input = turn['candidate_input'].lower()
    
    # Add your custom criteria here
    if "custom_keyword" in candidate_input:
        return "expected_response" in response
        
    # ... existing logic
```

### Custom Agent Detection

Modify `_analyze_agent_usage()` to improve agent usage detection:

```python
def _analyze_agent_usage(self, response: str, result: EvaluationResult):
    """Enhanced agent detection"""
    # Add more sophisticated keyword analysis
    # Use ML models for classification
    # Analyze log files for actual agent calls
```

## Output Files

The evaluation generates several output files:

- **`evaluation_report.md`**: Comprehensive markdown report
- **`evaluation_results.log`**: Detailed execution logs  
- **`evaluation_metrics.json`** (optional): Machine-readable results

## Performance Considerations

- **Memory**: Loads all conversations into memory
- **Speed**: ~1-3 seconds per conversation evaluation  
- **Logs**: Generates detailed logs for debugging
- **Scalability**: Tested with up to 100 conversations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the app directory
2. **JSON Format**: Validate your JSON structure matches expected format
3. **Missing Dependencies**: Install required packages from requirements.txt
4. **App Components**: Ensure main.py components are accessible

### Debug Tips

1. **Enable Verbose Logging**: Use `--verbose` flag
2. **Test Small Batches**: Use `--conversations 3` for quick testing
3. **Check Logs**: Review `evaluation_results.log` for details
4. **Validate JSON**: Use online JSON validators

## Example Output

After running evaluation, you'll see:

```
🎯 Conversation Evaluation Tool
========================================
JSON File: sms_conversations.json
Report File: evaluation_report.md  
Conversations: All
========================================
✅ Loaded 15 conversations from sms_conversations.json
🚀 Starting evaluation of 15 conversations
🎯 Evaluating conversation 1
✅ Conversation 1 evaluation complete - Accuracy: 0.80
...
============================================================
📊 EVALUATION RESULTS SUMMARY
============================================================
📈 Total Conversations: 15
🎯 Overall Accuracy: 78.3%
🔚 End Prediction Accuracy: 86.7%
📊 Confusion Matrix:
   True Positives:  5
   True Negatives:  8  
   False Positives: 1
   False Negatives: 1
📄 Full report saved to: evaluation_report.md
============================================================
```

This evaluation system provides comprehensive insights into your conversation assistant's performance and helps identify areas for improvement! 