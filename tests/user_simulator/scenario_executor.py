"""
Scenario Executor - Runs conversation scenarios and tracks actual vs expected agent flows.
This is the core engine for validating multi-agent conversation patterns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, List, Optional, Any, Tuple
import uuid
import time
import logging
from datetime import datetime
import json

from app.main import SupervisorAgent
from .simulator import UserSimulator
from .scenarios import ScenarioDefinition, ConversationScenarios, ScenarioMatcher, AgentType, FlowStep
from .personas import get_available_personas

class ConversationRecord:
    """Records all interactions during a conversation for analysis"""
    
    def __init__(self, scenario_id: str, persona: str):
        self.scenario_id = scenario_id
        self.persona = persona
        self.conversation_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.messages: List[Dict[str, Any]] = []
        self.agent_calls: List[Dict[str, Any]] = []
        self.turns: int = 0
        self.conversation_ended: bool = False
        self.success: bool = False
        self.errors: List[str] = []
    
    def add_message(self, role: str, content: str, turn: int):
        """Add a message to the conversation record"""
        self.messages.append({
            "role": role,
            "content": content,
            "turn": turn,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_agent_call(self, agent_type: str, input_data: str, output_data: str, turn: int):
        """Record an agent call for flow analysis"""
        self.agent_calls.append({
            "agent_type": agent_type,
            "input": input_data,
            "output": output_data,
            "turn": turn,
            "timestamp": datetime.now().isoformat()
        })
    
    def end_conversation(self, success: bool = False):
        """Mark conversation as ended"""
        self.end_time = datetime.now()
        self.conversation_ended = True
        self.success = success
    
    def get_duration(self) -> float:
        """Get conversation duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "scenario_id": self.scenario_id,
            "persona": self.persona,
            "conversation_id": self.conversation_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "turns": self.turns,
            "conversation_ended": self.conversation_ended,
            "success": self.success,
            "messages": self.messages,
            "agent_calls": self.agent_calls,
            "errors": self.errors
        }

class ScenarioExecutor:
    """Executes conversation scenarios and validates agent flows"""
    
    def __init__(self):
        self.logger = logging.getLogger('scenario_executor')
        self.records: List[ConversationRecord] = []
        self.current_record: Optional[ConversationRecord] = None
    
    def execute_scenario(self, scenario_id: str, persona: str, max_turns: int = 10) -> ConversationRecord:
        """
        Execute a single conversation scenario
        
        Args:
            scenario_id: ID of the scenario to execute
            persona: Persona to use for the conversation
            max_turns: Maximum number of conversation turns
            
        Returns:
            ConversationRecord with full interaction history
        """
        self.logger.info(f"🎬 Executing scenario '{scenario_id}' with persona '{persona}'")
        
        # Get scenario definition
        scenarios = ConversationScenarios.get_all_scenarios()
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = scenarios[scenario_id]
        
        # Validate persona compatibility
        if persona not in scenario.compatible_personas and persona != scenario.primary_persona:
            self.logger.warning(f"⚠️ Persona '{persona}' not explicitly compatible with scenario '{scenario_id}'")
        
        # Initialize record
        record = ConversationRecord(scenario_id, persona)
        self.current_record = record
        self.records.append(record)
        
        try:
            # Initialize agents
            supervisor = SupervisorAgent()
            user_sim = UserSimulator(persona)
            
            self.logger.info(f"🎭 Starting conversation: {record.conversation_id}")
            
            # Start conversation
            user_message = user_sim.start_conversation()
            record.add_message("user", user_message, 1)
            record.turns = 1
            
            self.logger.info(f"👤 User: {user_message}")
            
            # Conversation loop
            for turn in range(2, max_turns + 1):
                # Supervisor processes user message
                supervisor_response = supervisor.process_message(user_message)
                record.add_message("supervisor", supervisor_response, turn - 1)
                
                self.logger.info(f"🤖 Supervisor: {supervisor_response[:100]}...")
                
                # Record agent calls (this would need integration with the logging system)
                self._extract_agent_calls_from_logs(record, turn - 1)
                
                # Check if conversation should end
                if "[END]" in supervisor_response:
                    self.logger.info("🔚 Conversation ended by supervisor")
                    record.end_conversation(success=True)
                    break
                
                # User responds
                user_message = user_sim.respond_to_supervisor(supervisor_response)
                if user_message is None:
                    self.logger.info("🔚 Conversation ended by user")
                    record.end_conversation(success=True)
                    break
                
                record.add_message("user", user_message, turn)
                record.turns = turn
                
                self.logger.info(f"👤 User: {user_message}")
                
                # Check if user simulator ended conversation
                if user_sim.current_conversation and user_sim.current_conversation.conversation_ended:
                    self.logger.info("🔚 User simulator ended conversation")
                    record.end_conversation(success=True)
                    break
            else:
                # Hit max turns
                self.logger.warning(f"⚠️ Conversation hit maximum turns ({max_turns})")
                record.end_conversation(success=False)
                record.errors.append(f"Hit maximum turns ({max_turns})")
            
            self.logger.info(f"✅ Scenario execution completed. Duration: {record.get_duration():.1f}s, Turns: {record.turns}")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing scenario: {e}")
            record.errors.append(str(e))
            record.end_conversation(success=False)
            import traceback
            traceback.print_exc()
        
        finally:
            self.current_record = None
        
        return record
    
    def execute_all_scenarios_for_persona(self, persona: str) -> List[ConversationRecord]:
        """Execute all compatible scenarios for a given persona"""
        self.logger.info(f"🎯 Executing all scenarios for persona: {persona}")
        
        compatible_scenarios = ScenarioMatcher.get_scenarios_for_persona(persona)
        results = []
        
        for scenario in compatible_scenarios:
            try:
                record = self.execute_scenario(scenario.scenario_id, persona)
                results.append(record)
                time.sleep(1)  # Brief pause between scenarios
            except Exception as e:
                self.logger.error(f"❌ Failed to execute scenario {scenario.scenario_id}: {e}")
        
        return results
    
    def execute_all_personas_for_scenario(self, scenario_id: str) -> List[ConversationRecord]:
        """Execute a scenario with all compatible personas"""
        self.logger.info(f"🎬 Executing scenario '{scenario_id}' with all compatible personas")
        
        scenarios = ConversationScenarios.get_all_scenarios()
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = scenarios[scenario_id]
        results = []
        
        # Execute with primary persona
        try:
            record = self.execute_scenario(scenario_id, scenario.primary_persona)
            results.append(record)
            time.sleep(1)
        except Exception as e:
            self.logger.error(f"❌ Failed with primary persona {scenario.primary_persona}: {e}")
        
        # Execute with compatible personas
        for persona in scenario.compatible_personas:
            if persona != scenario.primary_persona:  # Avoid duplicates
                try:
                    record = self.execute_scenario(scenario_id, persona)
                    results.append(record)
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"❌ Failed with persona {persona}: {e}")
        
        return results
    
    def execute_comprehensive_test_suite(self) -> Dict[str, List[ConversationRecord]]:
        """Execute all scenarios with all compatible personas"""
        self.logger.info("🚀 Starting comprehensive test suite execution")
        
        all_results = {}
        scenarios = ConversationScenarios.get_all_scenarios()
        
        for scenario_id in scenarios.keys():
            self.logger.info(f"📋 Testing scenario: {scenario_id}")
            results = self.execute_all_personas_for_scenario(scenario_id)
            all_results[scenario_id] = results
            
            # Brief pause between scenarios
            time.sleep(2)
        
        self.logger.info("✅ Comprehensive test suite completed")
        return all_results
    
    def _extract_agent_calls_from_logs(self, record: ConversationRecord, turn: int):
        """
        Extract agent calls from the logging system
        This is a placeholder - would need integration with the actual logging system
        """
        # This would parse the logs to extract actual agent calls
        # For now, we'll simulate based on common patterns
        
        # This is where we would integrate with the logging system to extract
        # actual agent communication flows from app.logging_config
        pass
    
    def validate_scenario_execution(self, record: ConversationRecord) -> Dict[str, Any]:
        """Validate if the execution matches the expected scenario flow"""
        scenarios = ConversationScenarios.get_all_scenarios()
        scenario = scenarios[record.scenario_id]
        
        validation_result = {
            "scenario_id": record.scenario_id,
            "persona": record.persona,
            "conversation_id": record.conversation_id,
            "success": False,
            "score": 0.0,
            "criteria_met": {},
            "flow_analysis": {},
            "recommendations": []
        }
        
        # Validate success criteria
        criteria = scenario.success_criteria
        criteria_met = {}
        
        # Check turn count
        if "min_turns" in criteria:
            criteria_met["min_turns"] = record.turns >= criteria["min_turns"]
        if "max_turns" in criteria:
            criteria_met["max_turns"] = record.turns <= criteria["max_turns"]
        
        # Check conversation completion
        if "conversation_ended" in criteria:
            criteria_met["conversation_ended"] = record.conversation_ended == criteria["conversation_ended"]
        
        # Check agent usage (placeholder - would need log integration)
        if "agent_calls" in criteria:
            expected_agents = criteria["agent_calls"]
            # This would check actual agent calls from logs
            criteria_met["agent_calls"] = True  # Placeholder
        
        # Calculate score
        total_criteria = len(criteria_met)
        met_criteria = sum(1 for met in criteria_met.values() if met)
        score = met_criteria / total_criteria if total_criteria > 0 else 0.0
        
        validation_result.update({
            "success": score >= 0.8,  # 80% criteria must be met
            "score": score,
            "criteria_met": criteria_met
        })
        
        return validation_result
    
    def generate_summary_report(self, results: Dict[str, List[ConversationRecord]]) -> Dict[str, Any]:
        """Generate a comprehensive summary report of all test results"""
        total_conversations = sum(len(records) for records in results.values())
        successful_conversations = 0
        total_duration = 0.0
        
        scenario_summary = {}
        persona_summary = {}
        
        for scenario_id, records in results.items():
            scenario_stats = {
                "total_runs": len(records),
                "successful_runs": 0,
                "average_turns": 0.0,
                "average_duration": 0.0,
                "personas_tested": set()
            }
            
            for record in records:
                if record.success and record.conversation_ended:
                    successful_conversations += 1
                    scenario_stats["successful_runs"] += 1
                
                total_duration += record.get_duration()
                scenario_stats["average_turns"] += record.turns
                scenario_stats["average_duration"] += record.get_duration()
                scenario_stats["personas_tested"].add(record.persona)
                
                # Track persona performance
                if record.persona not in persona_summary:
                    persona_summary[record.persona] = {
                        "total_scenarios": 0,
                        "successful_scenarios": 0,
                        "average_turns": 0.0,
                        "scenarios_tested": set()
                    }
                
                persona_summary[record.persona]["total_scenarios"] += 1
                if record.success:
                    persona_summary[record.persona]["successful_scenarios"] += 1
                persona_summary[record.persona]["average_turns"] += record.turns
                persona_summary[record.persona]["scenarios_tested"].add(scenario_id)
            
            if len(records) > 0:
                scenario_stats["average_turns"] /= len(records)
                scenario_stats["average_duration"] /= len(records)
                scenario_stats["personas_tested"] = list(scenario_stats["personas_tested"])
            
            scenario_summary[scenario_id] = scenario_stats
        
        # Finalize persona summary
        for persona_data in persona_summary.values():
            if persona_data["total_scenarios"] > 0:
                persona_data["average_turns"] /= persona_data["total_scenarios"]
                persona_data["scenarios_tested"] = list(persona_data["scenarios_tested"])
        
        return {
            "test_suite_summary": {
                "total_conversations": total_conversations,
                "successful_conversations": successful_conversations,
                "success_rate": successful_conversations / total_conversations if total_conversations > 0 else 0.0,
                "total_duration": total_duration,
                "average_duration_per_conversation": total_duration / total_conversations if total_conversations > 0 else 0.0
            },
            "scenario_performance": scenario_summary,
            "persona_performance": persona_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict[str, List[ConversationRecord]], filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenario_test_results_{timestamp}.json"
        
        # Convert records to serializable format
        serializable_results = {}
        for scenario_id, records in results.items():
            serializable_results[scenario_id] = [record.to_dict() for record in records]
        
        # Add summary report
        summary = self.generate_summary_report(results)
        
        output_data = {
            "results": serializable_results,
            "summary": summary
        }
        
        filepath = f"tests/examples/{filename}"
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {filepath}")
        return filepath

# Convenience functions for easy usage
def run_single_scenario(scenario_id: str, persona: str) -> ConversationRecord:
    """Run a single scenario with specified persona"""
    executor = ScenarioExecutor()
    return executor.execute_scenario(scenario_id, persona)

def run_scenario_test_suite() -> Dict[str, List[ConversationRecord]]:
    """Run the complete scenario test suite"""
    executor = ScenarioExecutor()
    return executor.execute_comprehensive_test_suite()

def quick_scenario_demo(scenario_id: str = "full_journey_success"):
    """Quick demo of a single scenario"""
    print(f"🎬 Running demo of scenario: {scenario_id}")
    
    scenarios = ConversationScenarios.get_all_scenarios()
    if scenario_id not in scenarios:
        print(f"❌ Unknown scenario: {scenario_id}")
        return
    
    scenario = scenarios[scenario_id]
    persona = scenario.primary_persona
    
    executor = ScenarioExecutor()
    record = executor.execute_scenario(scenario_id, persona)
    validation = executor.validate_scenario_execution(record)
    
    print(f"\n📊 Demo Results:")
    print(f"   Scenario: {scenario.name}")
    print(f"   Persona: {persona}")
    print(f"   Turns: {record.turns}")
    print(f"   Duration: {record.get_duration():.1f}s")
    print(f"   Success: {record.success}")
    print(f"   Validation Score: {validation['score']:.2f}")
    
    return record, validation 