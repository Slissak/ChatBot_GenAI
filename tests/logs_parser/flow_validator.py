"""
Flow Validator - Compares expected scenario flows with actual agent communication patterns.
This system validates that conversations follow expected agent routing patterns and provides
detailed analysis of any discrepancies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

from .flow_extractor import ConversationFlow, ParsedAgentCall, FlowComparator
from tests.user_simulator.scenarios import ScenarioDefinition, ConversationScenarios, ExpectedFlow, FlowStep, AgentType

class ValidationResult:
    """Comprehensive validation result for a conversation flow"""
    
    def __init__(self, conversation_id: str, scenario_id: str):
        self.conversation_id = conversation_id
        self.scenario_id = scenario_id
        self.timestamp = datetime.now()
        
        # Core validation metrics
        self.overall_score: float = 0.0
        self.success: bool = False
        
        # Detailed analysis
        self.agent_flow_analysis: Dict[str, Any] = {}
        self.turn_analysis: Dict[str, Any] = {}
        self.timing_analysis: Dict[str, Any] = {}
        self.criteria_analysis: Dict[str, Any] = {}
        
        # Issues and recommendations
        self.critical_issues: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        
        # Raw data
        self.expected_flow: Optional[ScenarioDefinition] = None
        self.actual_flow: Optional[ConversationFlow] = None
        self.comparison_details: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary for JSON serialization"""
        return {
            "conversation_id": self.conversation_id,
            "scenario_id": self.scenario_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "success": self.success,
            "agent_flow_analysis": self.agent_flow_analysis,
            "turn_analysis": self.turn_analysis,
            "timing_analysis": self.timing_analysis,
            "criteria_analysis": self.criteria_analysis,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "comparison_details": self.comparison_details
        }

    def add_critical_issue(self, issue: str):
        """Add a critical issue that affects scenario success"""
        self.critical_issues.append(issue)
        self.success = False

    def add_warning(self, warning: str):
        """Add a warning that doesn't necessarily fail the scenario"""
        self.warnings.append(warning)

    def add_recommendation(self, recommendation: str):
        """Add a recommendation for improvement"""
        self.recommendations.append(recommendation)

class FlowValidator:
    """Advanced validator that compares expected vs actual conversation flows"""
    
    def __init__(self):
        self.logger = logging.getLogger('flow_validator')
        self.comparator = FlowComparator()
    
    def validate_conversation_flow(self, 
                                 actual_flow: ConversationFlow,
                                 expected_scenario: ScenarioDefinition) -> ValidationResult:
        """
        Comprehensively validate an actual conversation flow against expected scenario
        
        Args:
            actual_flow: The actual conversation flow extracted from logs
            expected_scenario: The expected scenario definition
            
        Returns:
            ValidationResult with detailed analysis
        """
        result = ValidationResult(actual_flow.conversation_id, expected_scenario.scenario_id)
        result.expected_flow = expected_scenario
        result.actual_flow = actual_flow
        
        self.logger.info(f"🔍 Validating conversation {actual_flow.conversation_id} against scenario {expected_scenario.scenario_id}")
        
        # Perform all validation checks
        self._validate_agent_usage(actual_flow, expected_scenario, result)
        self._validate_turn_count(actual_flow, expected_scenario, result)
        self._validate_conversation_completion(actual_flow, expected_scenario, result)
        self._validate_agent_sequence(actual_flow, expected_scenario, result)
        self._validate_success_criteria(actual_flow, expected_scenario, result)
        self._validate_timing_patterns(actual_flow, expected_scenario, result)
        
        # Calculate overall score and success
        self._calculate_overall_score(result)
        
        self.logger.info(f"✅ Validation complete. Score: {result.overall_score:.2f}, Success: {result.success}")
        
        return result
    
    def _validate_agent_usage(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate which agents were used vs expected"""
        expected_agents = self._extract_expected_agents(expected_scenario)
        actual_agents = actual_flow.get_unique_agents_used()
        
        result.agent_flow_analysis = {
            "expected_agents": expected_agents,
            "actual_agents": actual_agents,
            "correctly_used": [],
            "missing_agents": [],
            "unexpected_agents": [],
            "agent_call_counts": actual_flow.to_dict().get("agent_call_counts", {})
        }
        
        # Check for correctly used agents
        for agent in expected_agents:
            if agent in actual_agents:
                result.agent_flow_analysis["correctly_used"].append(agent)
            else:
                result.agent_flow_analysis["missing_agents"].append(agent)
                result.add_critical_issue(f"Expected agent '{agent}' was not used")
        
        # Check for unexpected agents
        for agent in actual_agents:
            if agent not in expected_agents:
                result.agent_flow_analysis["unexpected_agents"].append(agent)
                result.add_warning(f"Unexpected agent '{agent}' was used")
        
        # Special validation for specific criteria
        criteria = expected_scenario.success_criteria
        
        if criteria.get("no_info_agent") and "info_agent" in actual_agents:
            result.add_critical_issue("info_agent was used when scenario expected no info agent usage")
        
        if criteria.get("no_scheduling") and "sched_agent" in actual_agents:
            result.add_critical_issue("sched_agent was used when scenario expected no scheduling")
        
        if criteria.get("all_agents_used"):
            required_agents = ["info_agent", "sched_agent", "exit_agent"]
            missing_required = [agent for agent in required_agents if agent not in actual_agents]
            if missing_required:
                result.add_critical_issue(f"Scenario requires all agents but missing: {missing_required}")
    
    def _validate_turn_count(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate conversation turn count against expectations"""
        min_turns, max_turns = expected_scenario.estimated_turns
        actual_turns = actual_flow.total_turns
        
        result.turn_analysis = {
            "expected_range": {"min": min_turns, "max": max_turns},
            "actual_turns": actual_turns,
            "within_range": min_turns <= actual_turns <= max_turns,
            "variance": {
                "below_min": max(0, min_turns - actual_turns),
                "above_max": max(0, actual_turns - max_turns)
            }
        }
        
        if not result.turn_analysis["within_range"]:
            if actual_turns < min_turns:
                result.add_critical_issue(f"Conversation too short: {actual_turns} turns (expected {min_turns}-{max_turns})")
            else:
                result.add_warning(f"Conversation longer than expected: {actual_turns} turns (expected {min_turns}-{max_turns})")
        
        # Check for specific turn criteria
        criteria = expected_scenario.success_criteria
        
        if "min_turns" in criteria and actual_turns < criteria["min_turns"]:
            result.add_critical_issue(f"Conversation below minimum turns: {actual_turns} < {criteria['min_turns']}")
        
        if "max_turns" in criteria and actual_turns > criteria["max_turns"]:
            result.add_critical_issue(f"Conversation exceeded maximum turns: {actual_turns} > {criteria['max_turns']}")
    
    def _validate_conversation_completion(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate conversation completion status"""
        criteria = expected_scenario.success_criteria
        
        if "conversation_ended" in criteria:
            expected_ended = criteria["conversation_ended"]
            actual_ended = actual_flow.conversation_ended
            
            if expected_ended != actual_ended:
                if expected_ended:
                    result.add_critical_issue("Conversation should have ended but didn't")
                else:
                    result.add_warning("Conversation ended unexpectedly")
        
        # Check for successful outcomes
        if criteria.get("successful_scheduling"):
            # Look for scheduling-related patterns in agent calls
            has_scheduling = "sched_agent" in actual_flow.get_unique_agents_used()
            if not has_scheduling:
                result.add_critical_issue("Scenario expected successful scheduling but no scheduling agent was used")
    
    def _validate_agent_sequence(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate the sequence of agent calls"""
        actual_sequence = actual_flow.get_agent_sequence()
        expected_flows = expected_scenario.expected_flow
        
        result.agent_flow_analysis["agent_sequence"] = {
            "actual_sequence": actual_sequence,
            "sequence_analysis": [],
            "flow_violations": []
        }
        
        # Check for logical flow patterns
        agent_positions = {}
        for i, agent in enumerate(actual_sequence):
            if agent not in agent_positions:
                agent_positions[agent] = []
            agent_positions[agent].append(i)
        
        # Validate common flow patterns
        if "info_agent" in agent_positions and "sched_agent" in agent_positions:
            info_positions = agent_positions["info_agent"]
            sched_positions = agent_positions["sched_agent"]
            
            # Generally, info should come before scheduling
            if any(info_pos > sched_pos for info_pos in info_positions for sched_pos in sched_positions):
                result.add_warning("Information gathering occurred after scheduling started")
        
        # Check if exit_agent is typically called last
        if "exit_agent" in agent_positions and len(actual_sequence) > 1:
            last_agent = actual_sequence[-1]
            if last_agent != "exit_agent":
                result.add_warning("exit_agent was not the final agent called")
    
    def _validate_success_criteria(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate all success criteria from the scenario"""
        criteria = expected_scenario.success_criteria
        
        result.criteria_analysis = {
            "total_criteria": len(criteria),
            "met_criteria": 0,
            "failed_criteria": [],
            "criteria_details": {}
        }
        
        for criterion, expected_value in criteria.items():
            met = self._check_individual_criterion(actual_flow, criterion, expected_value, result)
            
            result.criteria_analysis["criteria_details"][criterion] = {
                "expected": expected_value,
                "met": met,
                "description": self._get_criterion_description(criterion)
            }
            
            if met:
                result.criteria_analysis["met_criteria"] += 1
            else:
                result.criteria_analysis["failed_criteria"].append(criterion)
    
    def _validate_timing_patterns(self, actual_flow: ConversationFlow, expected_scenario: ScenarioDefinition, result: ValidationResult):
        """Validate timing patterns and efficiency"""
        result.timing_analysis = {
            "total_duration": actual_flow.duration_seconds,
            "average_time_per_turn": 0.0,
            "agent_response_times": {},
            "efficiency_rating": "unknown"
        }
        
        if actual_flow.total_turns > 0:
            result.timing_analysis["average_time_per_turn"] = actual_flow.duration_seconds / actual_flow.total_turns
        
        # Analyze efficiency based on scenario tags
        scenario_tags = expected_scenario.tags
        
        if "quick" in scenario_tags or "fast" in scenario_tags or "efficient" in scenario_tags:
            if actual_flow.duration_seconds > 60:  # More than 1 minute for quick scenarios
                result.add_warning(f"Scenario tagged as 'quick' but took {actual_flow.duration_seconds:.1f} seconds")
                result.timing_analysis["efficiency_rating"] = "slow"
            else:
                result.timing_analysis["efficiency_rating"] = "good"
        
        elif "thorough" in scenario_tags or "detailed" in scenario_tags:
            if actual_flow.duration_seconds < 30:  # Less than 30 seconds for thorough scenarios
                result.add_warning(f"Scenario tagged as 'thorough' but only took {actual_flow.duration_seconds:.1f} seconds")
                result.timing_analysis["efficiency_rating"] = "too_fast"
            else:
                result.timing_analysis["efficiency_rating"] = "appropriate"
    
    def _check_individual_criterion(self, actual_flow: ConversationFlow, criterion: str, expected_value: Any, result: ValidationResult) -> bool:
        """Check a single success criterion"""
        if criterion == "expressed_interest":
            # This would require semantic analysis of conversation content
            # For now, we'll use proxy indicators
            return len(actual_flow.agent_calls) > 2  # More engagement suggests interest
        
        elif criterion == "expressed_disinterest":
            # Quick conversations with few agent calls suggest disinterest
            return actual_flow.total_turns <= 3 and len(actual_flow.get_unique_agents_used()) <= 2
        
        elif criterion == "scheduling_discussed":
            return "sched_agent" in actual_flow.get_unique_agents_used()
        
        elif criterion == "info_agent_used":
            return "info_agent" in actual_flow.get_unique_agents_used()
        
        elif criterion == "fast_scheduling":
            return ("sched_agent" in actual_flow.get_unique_agents_used() and 
                    actual_flow.total_turns <= 4)
        
        elif criterion == "immediate_rejection":
            return (actual_flow.total_turns <= 2 and 
                    len(actual_flow.get_unique_agents_used()) <= 1)
        
        elif criterion == "multiple_info_calls":
            return actual_flow.get_agent_call_count("info_agent") >= 2
        
        elif criterion == "conversation_ended":
            return actual_flow.conversation_ended == expected_value
        
        # Default handling for boolean criteria
        return bool(expected_value)
    
    def _get_criterion_description(self, criterion: str) -> str:
        """Get human-readable description of a criterion"""
        descriptions = {
            "expressed_interest": "User showed interest in the position",
            "expressed_disinterest": "User clearly indicated lack of interest",
            "scheduling_discussed": "Interview scheduling was discussed",
            "info_agent_used": "Information agent was used to answer questions",
            "fast_scheduling": "Scheduling occurred quickly without extensive discussion",
            "immediate_rejection": "User rejected opportunity immediately",
            "multiple_info_calls": "Information agent was called multiple times",
            "conversation_ended": "Conversation reached proper conclusion",
            "successful_scheduling": "Interview was successfully scheduled",
            "no_info_agent": "Information agent should not be used",
            "no_scheduling": "Scheduling should not occur",
            "all_agents_used": "All available agents should be utilized"
        }
        return descriptions.get(criterion, f"Check for {criterion}")
    
    def _calculate_overall_score(self, result: ValidationResult):
        """Calculate overall validation score"""
        # Weight different aspects of validation
        weights = {
            "agent_usage": 0.4,
            "turn_count": 0.2,
            "criteria": 0.3,
            "completion": 0.1
        }
        
        # Agent usage score
        expected_count = len(result.agent_flow_analysis.get("expected_agents", []))
        correct_count = len(result.agent_flow_analysis.get("correctly_used", []))
        unexpected_count = len(result.agent_flow_analysis.get("unexpected_agents", []))
        
        agent_score = 0.0
        if expected_count > 0:
            agent_score = correct_count / expected_count
            # Penalize unexpected agents
            if unexpected_count > 0:
                agent_score = max(0.0, agent_score - (unexpected_count * 0.1))
        
        # Turn count score
        turn_score = 1.0 if result.turn_analysis.get("within_range", False) else 0.5
        
        # Criteria score
        total_criteria = result.criteria_analysis.get("total_criteria", 1)
        met_criteria = result.criteria_analysis.get("met_criteria", 0)
        criteria_score = met_criteria / total_criteria if total_criteria > 0 else 0.0
        
        # Completion score
        completion_score = 1.0 if len(result.critical_issues) == 0 else 0.0
        
        # Calculate weighted score
        result.overall_score = (
            agent_score * weights["agent_usage"] +
            turn_score * weights["turn_count"] +
            criteria_score * weights["criteria"] +
            completion_score * weights["completion"]
        )
        
        # Determine success (80% threshold)
        result.success = result.overall_score >= 0.8 and len(result.critical_issues) == 0
        
        # Add final recommendations
        if result.overall_score < 0.8:
            result.add_recommendation(f"Overall score {result.overall_score:.2f} below success threshold of 0.8")
        
        if len(result.critical_issues) > 0:
            result.add_recommendation(f"Address {len(result.critical_issues)} critical issues")
        
        if len(result.warnings) > 0:
            result.add_recommendation(f"Consider addressing {len(result.warnings)} warnings")
    
    def _extract_expected_agents(self, scenario: ScenarioDefinition) -> List[str]:
        """Extract list of expected agents from scenario"""
        expected_agents = []
        
        # From success criteria
        if "agent_calls" in scenario.success_criteria:
            agent_calls = scenario.success_criteria["agent_calls"]
            if isinstance(agent_calls, list):
                expected_agents.extend(agent_calls)
        
        # From expected flow (if any agent calls are specified)
        for flow_step in scenario.expected_flow:
            if flow_step.step_type == FlowStep.AGENT_CALL and flow_step.agent_type:
                agent_name = flow_step.agent_type.value  # Convert enum to string
                if agent_name not in expected_agents:
                    expected_agents.append(agent_name)
        
        return expected_agents

# Convenience functions for batch validation
def validate_scenario_execution_batch(conversations: List[ConversationFlow], 
                                    scenario_id: str) -> List[ValidationResult]:
    """Validate multiple conversations against the same scenario"""
    scenarios = ConversationScenarios.get_all_scenarios()
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    
    scenario = scenarios[scenario_id]
    validator = FlowValidator()
    
    results = []
    for conversation in conversations:
        result = validator.validate_conversation_flow(conversation, scenario)
        results.append(result)
    
    return results

def generate_validation_report(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Generate comprehensive validation report from multiple results"""
    if not validation_results:
        return {"error": "No validation results provided"}
    
    total_validations = len(validation_results)
    successful_validations = sum(1 for result in validation_results if result.success)
    
    # Aggregate statistics
    total_score = sum(result.overall_score for result in validation_results)
    average_score = total_score / total_validations
    
    # Issue analysis
    all_critical_issues = []
    all_warnings = []
    
    for result in validation_results:
        all_critical_issues.extend(result.critical_issues)
        all_warnings.extend(result.warnings)
    
    # Common issues
    issue_frequency = {}
    for issue in all_critical_issues:
        issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
    
    return {
        "summary": {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations,
            "average_score": average_score,
            "timestamp": datetime.now().isoformat()
        },
        "issue_analysis": {
            "total_critical_issues": len(all_critical_issues),
            "total_warnings": len(all_warnings),
            "common_issues": sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        },
        "individual_results": [result.to_dict() for result in validation_results]
    } 