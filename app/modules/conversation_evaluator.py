"""
Conversation Evaluation Module

This module reads conversation JSON files, simulates conversations with the app,
and evaluates the accuracy of the supervisor agent's responses and decision-making.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

# Add parent directory to path to import main app components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    conversation_id: int
    total_turns: int
    correct_predictions: int
    accuracy: float
    expected_end: str
    actual_end: str
    conversation_flow: List[str]
    errors: List[str]
    agent_usage: Dict[str, int]
    timing_info: Dict[str, float]

@dataclass
class OverallMetrics:
    """Overall evaluation metrics across all conversations"""
    total_conversations: int
    total_accuracy: float
    end_prediction_accuracy: float
    confusion_matrix: Dict[str, int]
    agent_usage_stats: Dict[str, int]
    common_errors: List[str]
    conversation_results: List[EvaluationResult]

class ConversationEvaluator:
    """
    Main evaluation class that reads JSON conversations and evaluates app performance
    """
    
    def __init__(self, json_file_path: str, app_instance=None):
        """
        Initialize the evaluator
        
        Args:
            json_file_path: Path to the conversation JSON file
            app_instance: Optional app instance for testing (if None, will import)
        """
        self.json_file_path = json_file_path
        self.conversations = []
        self.app_instance = app_instance
        self.results = []
        
        # Setup logging
        self.logger = logging.getLogger('conversation_evaluator')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        )
        
        # Create file handler
        handler = logging.FileHandler('evaluation_results.log')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def load_conversations(self) -> bool:
        """Load conversations from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                self.conversations = json.load(file)
            
            self.logger.info(f"✅ Loaded {len(self.conversations)} conversations from {self.json_file_path}")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {self.json_file_path}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading conversations: {e}")
            return False
    
    def _import_app_components(self):
        """Import app components if not provided"""
        if self.app_instance is None:
            try:
                from main import SupervisorAgent
                self.supervisor_class = SupervisorAgent
                self.logger.info("✅ Successfully imported app components")
                return True
            except ImportError as e:
                self.logger.error(f"❌ Failed to import app components: {e}")
                return False
        return True
    
    def _simulate_conversation(self, conversation: Dict) -> EvaluationResult:
        """
        Simulate a single conversation and evaluate results
        
        Args:
            conversation: Single conversation dictionary from JSON
            
        Returns:
            EvaluationResult with evaluation metrics
        """
        conv_id = conversation['conversation_id']
        turns = conversation['turns']
        
        self.logger.info(f"🎯 Evaluating conversation {conv_id}")
        
        # Initialize result tracking
        result = EvaluationResult(
            conversation_id=conv_id,
            total_turns=len(turns),
            correct_predictions=0,
            accuracy=0.0,
            expected_end="",
            actual_end="",
            conversation_flow=[],
            errors=[],
            agent_usage={"info_agent": 0, "sched_agent": 0, "exit_agent": 0},
            timing_info={}
        )
        
        try:
            # Create supervisor instance for this conversation
            if self.app_instance:
                supervisor = self.app_instance
            else:
                supervisor = self.supervisor_class()
            
            # Track conversation flow and timing
            start_time = datetime.now()
            
            # Simulate each turn in the conversation
            for turn in turns:
                if turn['speaker'] == 'candidate':
                    candidate_message = turn['text']
                    
                    # Process message through supervisor
                    turn_start = datetime.now()
                    try:
                        response = supervisor.process_message(candidate_message)
                        turn_duration = (datetime.now() - turn_start).total_seconds()
                        
                        # Track response and analyze
                        result.conversation_flow.append({
                            'turn_id': turn['turn_id'],
                            'candidate_input': candidate_message,
                            'supervisor_response': response,
                            'duration_seconds': turn_duration,
                            'expected_label': turn.get('label'),
                        })
                        
                        # Analyze which agents were likely used (based on response content)
                        self._analyze_agent_usage(response, result)
                        
                        # Check for end condition
                        if response.strip() == "[END]":
                            result.actual_end = "END"
                            break
                            
                    except Exception as e:
                        result.errors.append(f"Turn {turn['turn_id']}: {str(e)}")
                        self.logger.error(f"❌ Error in turn {turn['turn_id']}: {e}")
            
            # Determine expected end condition
            last_recruiter_turn = None
            for turn in reversed(turns):
                if turn['speaker'] == 'recruiter' and 'label' in turn and turn['label']:
                    last_recruiter_turn = turn
                    break
            
            if last_recruiter_turn and last_recruiter_turn['label'] == 'end':
                result.expected_end = "END"
            else:
                result.expected_end = "NOT_END"
            
            # If conversation didn't end naturally, check exit agent
            if not result.actual_end:
                try:
                    # Build conversation history for exit agent
                    conv_history = self._build_conversation_history(result.conversation_flow)
                    exit_decision = self._test_exit_agent(supervisor, conv_history)
                    result.actual_end = exit_decision
                except Exception as e:
                    result.errors.append(f"Exit agent test failed: {str(e)}")
                    result.actual_end = "NOT_END"
            
            # Calculate accuracy metrics
            result.accuracy = self._calculate_turn_accuracy(result.conversation_flow)
            result.correct_predictions = len([t for t in result.conversation_flow 
                                            if self._is_turn_correct(t)])
            
            # Calculate timing
            total_duration = (datetime.now() - start_time).total_seconds()
            result.timing_info = {
                'total_duration': total_duration,
                'avg_turn_duration': total_duration / len(result.conversation_flow) if result.conversation_flow else 0
            }
            
            self.logger.info(f"✅ Conversation {conv_id} evaluation complete - Accuracy: {result.accuracy:.2f}")
            
        except Exception as e:
            result.errors.append(f"Critical error: {str(e)}")
            self.logger.error(f"❌ Critical error evaluating conversation {conv_id}: {e}")
        
        return result
    
    def _analyze_agent_usage(self, response: str, result: EvaluationResult):
        """Analyze response to determine which agents were likely used"""
        response_lower = response.lower()
        
        # Info agent indicators
        if any(keyword in response_lower for keyword in ['python', 'sql', 'experience', 'technology', 'programming']):
            result.agent_usage['info_agent'] += 1
        
        # Scheduling agent indicators  
        if any(keyword in response_lower for keyword in ['schedule', 'interview', 'time', 'date', 'available', 'slot']):
            result.agent_usage['sched_agent'] += 1
        
        # Exit agent indicators
        if '[END]' in response or any(keyword in response_lower for keyword in ['goodbye', 'take care', 'end']):
            result.agent_usage['exit_agent'] += 1
    
    def _build_conversation_history(self, conversation_flow: List[Dict]) -> str:
        """Build conversation history string for exit agent testing"""
        history = ""
        for turn in conversation_flow:
            history += f"Candidate: {turn['candidate_input']}\n"
            history += f"Recruiter: {turn['supervisor_response']}\n\n"
        return history.strip()
    
    def _test_exit_agent(self, supervisor, conversation_history: str) -> str:
        """Test exit agent decision"""
        try:
            # Access exit agent from supervisor
            exit_agent = supervisor.exit_agent
            
            # Use the exit agent's analyze_conversation_ending tool
            result = exit_agent.agent.tools[0].func(conversation_history)
            
            return "END" if "END" in result else "NOT_END"
            
        except Exception as e:
            self.logger.error(f"❌ Exit agent test failed: {e}")
            return "NOT_END"
    
    def _calculate_turn_accuracy(self, conversation_flow: List[Dict]) -> float:
        """Calculate accuracy based on appropriate responses per turn"""
        if not conversation_flow:
            return 0.0
        
        correct_turns = sum(1 for turn in conversation_flow if self._is_turn_correct(turn))
        return correct_turns / len(conversation_flow)
    
    def _is_turn_correct(self, turn: Dict) -> bool:
        """
        Determine if a turn response was appropriate
        This is a simplified heuristic - you can make it more sophisticated
        """
        response = turn['supervisor_response'].lower()
        candidate_input = turn['candidate_input'].lower()
        
        # Check for basic appropriateness
        if len(response.strip()) == 0:
            return False
        
        # If candidate mentions scheduling, response should mention scheduling
        if any(word in candidate_input for word in ['schedule', 'interview', 'time', 'date']):
            return any(word in response for word in ['schedule', 'interview', 'time', 'available', 'slot'])
        
        # If candidate asks about job, response should be informative
        if any(word in candidate_input for word in ['python', 'experience', 'technology', 'job', 'position']):
            return len(response) > 50 and not 'error' in response.lower()
        
        # If candidate seems uninterested, should not push too hard
        if any(phrase in candidate_input for phrase in ['not interested', 'remove me', 'stop']):
            return any(word in response for word in ['understand', 'no problem', 'take care', '[END]'])
        
        # Default: if response is substantial and doesn't contain error
        return len(response) > 20 and 'error' not in response.lower()
    
    def evaluate_all_conversations(self) -> OverallMetrics:
        """
        Evaluate all conversations and return comprehensive metrics
        
        Returns:
            OverallMetrics with complete evaluation results
        """
        if not self.conversations:
            self.logger.error("❌ No conversations loaded")
            return None
        
        if not self._import_app_components():
            return None
        
        self.logger.info(f"🚀 Starting evaluation of {len(self.conversations)} conversations")
        
        # Evaluate each conversation
        for i, conversation in enumerate(self.conversations, 1):
            self.logger.info(f"📊 Processing conversation {i}/{len(self.conversations)}")
            result = self._simulate_conversation(conversation)
            self.results.append(result)
        
        # Calculate overall metrics
        return self._calculate_overall_metrics()
    
    def _calculate_overall_metrics(self) -> OverallMetrics:
        """Calculate comprehensive metrics across all conversations"""
        
        # Basic stats
        total_conversations = len(self.results)
        total_accuracy = sum(r.accuracy for r in self.results) / total_conversations if total_conversations > 0 else 0
        
        # End prediction accuracy (confusion matrix)
        confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        
        for result in self.results:
            expected = result.expected_end
            actual = result.actual_end
            
            if expected == "END" and actual == "END":
                confusion_matrix["TP"] += 1
            elif expected == "NOT_END" and actual == "NOT_END":
                confusion_matrix["TN"] += 1
            elif expected == "NOT_END" and actual == "END":
                confusion_matrix["FP"] += 1
            elif expected == "END" and actual == "NOT_END":
                confusion_matrix["FN"] += 1
        
        # End prediction accuracy
        correct_end_predictions = confusion_matrix["TP"] + confusion_matrix["TN"]
        end_prediction_accuracy = correct_end_predictions / total_conversations if total_conversations > 0 else 0
        
        # Agent usage stats
        agent_usage_stats = {"info_agent": 0, "sched_agent": 0, "exit_agent": 0}
        for result in self.results:
            for agent, count in result.agent_usage.items():
                agent_usage_stats[agent] += count
        
        # Common errors
        all_errors = []
        for result in self.results:
            all_errors.extend(result.errors)
        common_errors = list(set(all_errors))
        
        return OverallMetrics(
            total_conversations=total_conversations,
            total_accuracy=total_accuracy,
            end_prediction_accuracy=end_prediction_accuracy,
            confusion_matrix=confusion_matrix,
            agent_usage_stats=agent_usage_stats,
            common_errors=common_errors,
            conversation_results=self.results
        )
    
    def generate_report(self, metrics: OverallMetrics, output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            metrics: OverallMetrics object with evaluation results
            output_file: Optional file path to save report
            
        Returns:
            Report string
        """
        report = f"""
# Conversation Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance Summary
- **Total Conversations Evaluated**: {metrics.total_conversations}
- **Average Response Accuracy**: {metrics.total_accuracy:.1%}
- **End Prediction Accuracy**: {metrics.end_prediction_accuracy:.1%}

## Confusion Matrix for End Prediction
```
                 Predicted
              NOT_END   END
Actual NOT_END   {metrics.confusion_matrix['TN']:2d}     {metrics.confusion_matrix['FP']:2d}
       END       {metrics.confusion_matrix['FN']:2d}     {metrics.confusion_matrix['TP']:2d}
```

### End Prediction Metrics:
- **Precision**: {metrics.confusion_matrix['TP'] / (metrics.confusion_matrix['TP'] + metrics.confusion_matrix['FP']) if (metrics.confusion_matrix['TP'] + metrics.confusion_matrix['FP']) > 0 else 0:.1%}
- **Recall**: {metrics.confusion_matrix['TP'] / (metrics.confusion_matrix['TP'] + metrics.confusion_matrix['FN']) if (metrics.confusion_matrix['TP'] + metrics.confusion_matrix['FN']) > 0 else 0:.1%}

## Agent Usage Statistics
- **Info Agent Usage**: {metrics.agent_usage_stats['info_agent']} times
- **Scheduling Agent Usage**: {metrics.agent_usage_stats['sched_agent']} times  
- **Exit Agent Usage**: {metrics.agent_usage_stats['exit_agent']} times

## Detailed Conversation Results
"""
        
        for result in metrics.conversation_results:
            report += f"""
### Conversation {result.conversation_id}
- **Turns**: {result.total_turns}
- **Accuracy**: {result.accuracy:.1%}
- **Expected End**: {result.expected_end}
- **Actual End**: {result.actual_end}
- **Errors**: {len(result.errors)}
"""
            if result.errors:
                report += f"  - Error Details: {'; '.join(result.errors[:3])}\n"
        
        if metrics.common_errors:
            report += f"\n## Common Errors\n"
            for error in metrics.common_errors[:10]:  # Top 10 errors
                report += f"- {error}\n"
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📄 Report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save report: {e}")
        
        return report

def main():
    """Main function to run evaluation"""
    
    # Configuration
    JSON_FILE = "sms_conversations.json"
    REPORT_FILE = "evaluation_report.md"
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(JSON_FILE)
    
    # Load conversations
    if not evaluator.load_conversations():
        print("❌ Failed to load conversations")
        return
    
    # Run evaluation
    print("🚀 Starting conversation evaluation...")
    metrics = evaluator.evaluate_all_conversations()
    
    if metrics is None:
        print("❌ Evaluation failed")
        return
    
    # Generate and display report
    report = evaluator.generate_report(metrics, REPORT_FILE)
    
    print("\n" + "="*60)
    print("📊 EVALUATION COMPLETE")
    print("="*60)
    print(f"Total Conversations: {metrics.total_conversations}")
    print(f"Overall Accuracy: {metrics.total_accuracy:.1%}")
    print(f"End Prediction Accuracy: {metrics.end_prediction_accuracy:.1%}")
    print(f"Report saved to: {REPORT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main() 