"""
Conversation Scenarios - Defines all possible conversation flows with expected agent routing patterns.
Each scenario maps user persona to expected supervisor/agent interaction sequences.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import uuid
from datetime import datetime

class AgentType(Enum):
    """Types of agents in the system"""
    SUPERVISOR = "supervisor"
    INFO_AGENT = "info_agent"
    SCHED_AGENT = "sched_agent"
    EXIT_AGENT = "exit_agent"

class FlowStep(Enum):
    """Types of conversation flow steps"""
    USER_MESSAGE = "user_message"
    AGENT_CALL = "agent_call"
    AGENT_RESPONSE = "agent_response"
    CONVERSATION_END = "conversation_end"

@dataclass
class ExpectedFlow:
    """Represents an expected step in the conversation flow"""
    step_type: FlowStep
    agent_type: Optional[AgentType] = None
    description: str = ""
    turn_number: int = 0
    optional: bool = False  # Some steps might be optional depending on persona behavior

@dataclass
class ScenarioDefinition:
    """Complete definition of a conversation scenario"""
    scenario_id: str
    name: str
    description: str
    primary_persona: str  # Primary persona for this scenario
    compatible_personas: List[str]  # Other personas that could follow this flow
    expected_flow: List[ExpectedFlow]
    success_criteria: Dict[str, Any]
    estimated_turns: Tuple[int, int]  # (min_turns, max_turns)
    tags: List[str] = field(default_factory=list)

class ConversationScenarios:
    """Repository of all conversation scenarios for testing"""
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, ScenarioDefinition]:
        """Return all defined conversation scenarios"""
        return {
            "basic_greeting": ConversationScenarios.basic_greeting_scenario(),
            "info_query_interested": ConversationScenarios.info_query_interested_scenario(),
            "info_query_not_interested": ConversationScenarios.info_query_not_interested_scenario(),
            "full_journey_success": ConversationScenarios.full_journey_success_scenario(),
            "direct_scheduling": ConversationScenarios.direct_scheduling_scenario(),
            "immediate_rejection": ConversationScenarios.immediate_rejection_scenario(),
            "detailed_exploration": ConversationScenarios.detailed_exploration_scenario(),
            "indecisive_journey": ConversationScenarios.indecisive_journey_scenario(),
            "skeptical_evaluation": ConversationScenarios.skeptical_evaluation_scenario(),
            "quick_decision": ConversationScenarios.quick_decision_scenario()
        }
    
    @staticmethod
    def basic_greeting_scenario() -> ScenarioDefinition:
        """Scenario 1: Simple greeting exchange without deep engagement"""
        return ScenarioDefinition(
            scenario_id="basic_greeting",
            name="Basic Greeting Exchange",
            description="User greets, supervisor responds, minimal interaction before ending",
            primary_persona="disinterested",
            compatible_personas=["disinterested", "direct"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Initial greeting", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Supervisor processes greeting", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Brief response or decline", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Check if conversation should end", turn_number=2),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="Conversation ends quickly", turn_number=2)
            ],
            success_criteria={
                "max_turns": 3,
                "agent_calls": ["supervisor", "exit_agent"],
                "no_info_agent": True,
                "no_scheduling": True,
                "conversation_ended": True
            },
            estimated_turns=(2, 3),
            tags=["simple", "quick", "rejection"]
        )
    
    @staticmethod
    def info_query_interested_scenario() -> ScenarioDefinition:
        """Scenario 2: User asks about job, shows interest, but doesn't schedule"""
        return ScenarioDefinition(
            scenario_id="info_query_interested",
            name="Information Query - Interested",
            description="User asks about job role, gets information, shows interest but needs time to think",
            primary_persona="indecisive",
            compatible_personas=["indecisive", "detail_oriented", "skeptical"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Initial interest", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Supervisor handles greeting", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Ask about job details", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Retrieve job information", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Show interest but hesitate", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Check conversation status", turn_number=3),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="End with interest but no commitment", turn_number=4, optional=True)
            ],
            success_criteria={
                "min_turns": 3,
                "max_turns": 6,
                "agent_calls": ["supervisor", "info_agent", "exit_agent"],
                "info_agent_used": True,
                "expressed_interest": True,
                "no_scheduling": True
            },
            estimated_turns=(3, 6),
            tags=["information", "interested", "no_commitment"]
        )
    
    @staticmethod
    def info_query_not_interested_scenario() -> ScenarioDefinition:
        """Scenario 3: User asks about job but becomes uninterested after learning details"""
        return ScenarioDefinition(
            scenario_id="info_query_not_interested",
            name="Information Query - Not Interested",
            description="User asks about job, gets information, decides not interested",
            primary_persona="skeptical",
            compatible_personas=["skeptical", "direct"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Initial inquiry", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Supervisor responds", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Ask detailed questions", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Provide job details", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Express lack of interest", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Confirm conversation ending", turn_number=3),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="End due to lack of interest", turn_number=3)
            ],
            success_criteria={
                "min_turns": 3,
                "max_turns": 5,
                "agent_calls": ["supervisor", "info_agent", "exit_agent"],
                "info_agent_used": True,
                "expressed_disinterest": True,
                "conversation_ended": True
            },
            estimated_turns=(3, 5),
            tags=["information", "rejection", "after_details"]
        )
    
    @staticmethod
    def full_journey_success_scenario() -> ScenarioDefinition:
        """Scenario 4: Complete successful journey - greeting + questions + scheduling + conclusion"""
        return ScenarioDefinition(
            scenario_id="full_journey_success",
            name="Full Journey - Successful Scheduling",
            description="Complete conversation flow: greeting, job inquiry, scheduling, successful conclusion",
            primary_persona="eager",
            compatible_personas=["eager", "detail_oriented"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Enthusiastic greeting", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Welcome and introduce role", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Ask about job responsibilities", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Provide detailed job info", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Express interest and willingness to schedule", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SCHED_AGENT, "Check available interview slots", turn_number=3),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Confirm scheduling preference", turn_number=4),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Finalize scheduling details", turn_number=4),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Express gratitude", turn_number=5, optional=True),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Confirm successful conclusion", turn_number=5),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="End with successful interview scheduled", turn_number=5)
            ],
            success_criteria={
                "min_turns": 4,
                "max_turns": 8,
                "agent_calls": ["supervisor", "info_agent", "sched_agent", "exit_agent"],
                "all_agents_used": True,
                "expressed_interest": True,
                "scheduling_discussed": True,
                "successful_scheduling": True
            },
            estimated_turns=(4, 8),
            tags=["complete", "success", "scheduling", "all_agents"]
        )
    
    @staticmethod
    def direct_scheduling_scenario() -> ScenarioDefinition:
        """Scenario 5: Direct to scheduling without detailed job questions"""
        return ScenarioDefinition(
            scenario_id="direct_scheduling",
            name="Direct Scheduling Path",
            description="User shows immediate interest and moves directly to scheduling",
            primary_persona="direct",
            compatible_personas=["direct", "eager"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Direct interest expression", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Acknowledge interest", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Request to schedule immediately", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SCHED_AGENT, "Check available slots", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Confirm time slot", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Confirm completion", turn_number=3),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="Quick successful scheduling", turn_number=3)
            ],
            success_criteria={
                "max_turns": 4,
                "agent_calls": ["supervisor", "sched_agent", "exit_agent"],
                "no_info_agent": True,
                "fast_scheduling": True,
                "successful_scheduling": True
            },
            estimated_turns=(3, 4),
            tags=["direct", "fast", "scheduling", "efficient"]
        )
    
    @staticmethod
    def immediate_rejection_scenario() -> ScenarioDefinition:
        """Scenario 6: User immediately indicates not looking for job"""
        return ScenarioDefinition(
            scenario_id="immediate_rejection",
            name="Immediate Rejection",
            description="User politely declines immediately, not currently job searching",
            primary_persona="disinterested",
            compatible_personas=["disinterested"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Polite but clear disinterest", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Acknowledge response", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Confirm early termination", turn_number=1),
                ExpectedFlow(FlowStep.CONVERSATION_END, description="Professional conclusion", turn_number=1)
            ],
            success_criteria={
                "max_turns": 2,
                "agent_calls": ["supervisor", "exit_agent"],
                "immediate_rejection": True,
                "conversation_ended": True,
                "professional_closure": True
            },
            estimated_turns=(1, 2),
            tags=["rejection", "immediate", "professional"]
        )
    
    @staticmethod
    def detailed_exploration_scenario() -> ScenarioDefinition:
        """Extended scenario: Detail-oriented candidate asks many questions"""
        return ScenarioDefinition(
            scenario_id="detailed_exploration",
            name="Detailed Technical Exploration",
            description="Thorough candidate asks multiple detailed questions about role and company",
            primary_persona="detail_oriented",
            compatible_personas=["detail_oriented", "skeptical"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Initial detailed inquiry", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Welcome detailed candidate", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Technical stack questions", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Technical details", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Follow-up technical questions", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "More detailed responses", turn_number=3),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Company culture questions", turn_number=4),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Culture and process info", turn_number=4),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Express qualified interest", turn_number=5),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SCHED_AGENT, "Scheduling after thorough review", turn_number=5, optional=True),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Conclude detailed session", turn_number=6)
            ],
            success_criteria={
                "min_turns": 5,
                "max_turns": 10,
                "multiple_info_calls": True,
                "agent_calls": ["supervisor", "info_agent", "exit_agent"],
                "detailed_questions": True,
                "thorough_evaluation": True
            },
            estimated_turns=(5, 10),
            tags=["detailed", "technical", "thorough", "multiple_questions"]
        )
    
    @staticmethod
    def indecisive_journey_scenario() -> ScenarioDefinition:
        """Indecisive candidate changes mind multiple times"""
        return ScenarioDefinition(
            scenario_id="indecisive_journey",
            name="Indecisive Candidate Journey",
            description="Candidate shows interest, hesitates, asks for time to think",
            primary_persona="indecisive",
            compatible_personas=["indecisive"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Uncertain initial interest", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Encourage and inform", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Ask about qualifications needed", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Provide requirements", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Express self-doubt", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Provide reassurance", turn_number=3),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Request time to consider", turn_number=4),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Handle indecision", turn_number=4),
            ],
            success_criteria={
                "min_turns": 4,
                "max_turns": 7,
                "agent_calls": ["supervisor", "info_agent", "exit_agent"],
                "expressed_uncertainty": True,
                "requested_time": True,
                "no_immediate_scheduling": True
            },
            estimated_turns=(4, 7),
            tags=["indecisive", "uncertainty", "time_needed"]
        )
    
    @staticmethod
    def skeptical_evaluation_scenario() -> ScenarioDefinition:
        """Skeptical candidate thoroughly evaluates opportunity"""
        return ScenarioDefinition(
            scenario_id="skeptical_evaluation",
            name="Skeptical Candidate Evaluation",
            description="Experienced candidate carefully evaluates against current position",
            primary_persona="skeptical",
            compatible_personas=["skeptical"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Cautious initial response", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Address caution professionally", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Compensation and benefits inquiry", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Compensation details", turn_number=2),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Company stability questions", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Company information", turn_number=3),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Compare to current role", turn_number=4),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Address comparison", turn_number=4),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Conditional interest", turn_number=5),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SCHED_AGENT, "Explore scheduling", turn_number=5, optional=True),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Evaluate readiness", turn_number=6)
            ],
            success_criteria={
                "min_turns": 5,
                "max_turns": 8,
                "agent_calls": ["supervisor", "info_agent", "exit_agent"],
                "compensation_discussed": True,
                "comparison_made": True,
                "conditional_interest": True
            },
            estimated_turns=(5, 8),
            tags=["skeptical", "evaluation", "compensation", "comparison"]
        )
    
    @staticmethod
    def quick_decision_scenario() -> ScenarioDefinition:
        """Direct candidate makes quick decision (accept or reject)"""
        return ScenarioDefinition(
            scenario_id="quick_decision",
            name="Quick Decision Path",
            description="Time-pressed candidate makes rapid decision based on key factors",
            primary_persona="direct",
            compatible_personas=["direct"],
            expected_flow=[
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Direct inquiry about essentials", turn_number=1),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SUPERVISOR, "Provide key information", turn_number=1),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Quick follow-up on salary/remote", turn_number=2),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.INFO_AGENT, "Essential details only", turn_number=2, optional=True),
                ExpectedFlow(FlowStep.USER_MESSAGE, description="Immediate decision (yes/no)", turn_number=3),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.SCHED_AGENT, "Quick scheduling if yes", turn_number=3, optional=True),
                ExpectedFlow(FlowStep.AGENT_CALL, AgentType.EXIT_AGENT, "Conclude rapidly", turn_number=3)
            ],
            success_criteria={
                "max_turns": 4,
                "quick_decision": True,
                "essential_info_only": True,
                "efficient_process": True
            },
            estimated_turns=(2, 4),
            tags=["quick", "efficient", "decision", "direct"]
        )

class ScenarioMatcher:
    """Matches personas to appropriate scenarios and validates scenario execution"""
    
    @staticmethod
    def get_scenarios_for_persona(persona: str) -> List[ScenarioDefinition]:
        """Get all scenarios compatible with a given persona"""
        scenarios = ConversationScenarios.get_all_scenarios()
        compatible_scenarios = []
        
        for scenario in scenarios.values():
            if persona == scenario.primary_persona or persona in scenario.compatible_personas:
                compatible_scenarios.append(scenario)
        
        return compatible_scenarios
    
    @staticmethod
    def get_primary_scenario_for_persona(persona: str) -> Optional[ScenarioDefinition]:
        """Get the primary scenario for a persona"""
        scenarios = ConversationScenarios.get_all_scenarios()
        
        for scenario in scenarios.values():
            if persona == scenario.primary_persona:
                return scenario
        
        return None
    
    @staticmethod
    def validate_scenario_flow(scenario: ScenarioDefinition, actual_flow: List[Dict]) -> Dict[str, Any]:
        """Validate if actual conversation flow matches expected scenario"""
        validation_result = {
            "scenario_id": scenario.scenario_id,
            "success": False,
            "score": 0.0,
            "matched_steps": 0,
            "expected_steps": len(scenario.expected_flow),
            "missing_steps": [],
            "unexpected_steps": [],
            "criteria_met": {},
            "details": {}
        }
        
        # This will be implemented in the validation phase
        # For now, return structure for future implementation
        
        return validation_result 