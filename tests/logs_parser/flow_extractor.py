"""
Agent Flow Extractor - Parses logs to extract actual agent communication flows.
This system analyzes conversation logs to determine which agents were called,
in what order, and with what outcomes.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import os

# Import our scenario types for comparison
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tests.user_simulator.scenarios import AgentType, FlowStep

class LogEntryType(Enum):
    """Types of log entries we can parse"""
    AGENT_COMMUNICATION = "agent_communication"
    CONVERSATION_EVENT = "conversation_event"
    SYSTEM_HEALTH = "system_health"
    USER_MESSAGE = "user_message"
    SUPERVISOR_RESPONSE = "supervisor_response"

@dataclass
class ParsedAgentCall:
    """Represents a parsed agent call from logs"""
    timestamp: datetime
    conversation_id: str
    agent_name: str
    direction: str  # "to_agent" or "from_agent"
    message_content: str
    message_length: int
    status: str = "SUCCESS"
    turn_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "agent_name": self.agent_name,
            "direction": self.direction,
            "message_content": self.message_content,
            "message_length": self.message_length,
            "status": self.status,
            "turn_number": self.turn_number
        }

@dataclass
class ConversationFlow:
    """Represents the complete flow of a conversation"""
    conversation_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_turns: int
    agent_calls: List[ParsedAgentCall]
    user_messages: List[Dict[str, Any]]
    supervisor_responses: List[Dict[str, Any]]
    conversation_ended: bool = False
    duration_seconds: float = 0.0
    
    def get_agent_sequence(self) -> List[str]:
        """Get the sequence of agents called"""
        return [call.agent_name for call in self.agent_calls if call.direction == "to_agent"]
    
    def get_unique_agents_used(self) -> List[str]:
        """Get unique list of agents used in conversation"""
        return list(set(self.get_agent_sequence()))
    
    def get_agent_call_count(self, agent_name: str) -> int:
        """Count how many times a specific agent was called"""
        return len([call for call in self.agent_calls 
                   if call.agent_name == agent_name and call.direction == "to_agent"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_turns": self.total_turns,
            "agent_calls": [call.to_dict() for call in self.agent_calls],
            "user_messages": self.user_messages,
            "supervisor_responses": self.supervisor_responses,
            "conversation_ended": self.conversation_ended,
            "duration_seconds": self.duration_seconds,
            "agent_sequence": self.get_agent_sequence(),
            "unique_agents": self.get_unique_agents_used(),
            "agent_call_counts": {
                agent: self.get_agent_call_count(agent) 
                for agent in self.get_unique_agents_used()
            }
        }

class LogParser:
    """Parses various log formats to extract agent communication flows"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.logger = logging.getLogger('log_parser')
        
    def parse_json_logs(self, json_log_file: str = None) -> List[ConversationFlow]:
        """Parse structured JSON logs to extract conversation flows"""
        if json_log_file is None:
            json_log_file = os.path.join(self.log_dir, "app_structured.json")
        
        if not os.path.exists(json_log_file):
            self.logger.warning(f"JSON log file not found: {json_log_file}")
            return []
        
        conversations = {}
        
        try:
            with open(json_log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        log_entry = json.loads(line.strip())
                        self._process_json_log_entry(log_entry, conversations)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_num}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading JSON log file: {e}")
            return []
        
        # Convert to list and finalize conversations
        flows = list(conversations.values())
        for flow in flows:
            self._finalize_conversation_flow(flow)
        
        self.logger.info(f"Parsed {len(flows)} conversation flows from JSON logs")
        return flows
    
    def parse_agent_communication_logs(self, agent_log_file: str = None) -> List[ConversationFlow]:
        """Parse agent communication logs to extract flows"""
        if agent_log_file is None:
            agent_log_file = os.path.join(self.log_dir, "agent_communications.log")
        
        if not os.path.exists(agent_log_file):
            self.logger.warning(f"Agent communication log file not found: {agent_log_file}")
            return []
        
        conversations = {}
        
        try:
            with open(agent_log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        self._process_text_log_entry(line.strip(), conversations)
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_num}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading agent communication log file: {e}")
            return []
        
        # Convert to list and finalize conversations
        flows = list(conversations.values())
        for flow in flows:
            self._finalize_conversation_flow(flow)
        
        self.logger.info(f"Parsed {len(flows)} conversation flows from agent communication logs")
        return flows
    
    def _process_json_log_entry(self, log_entry: Dict[str, Any], conversations: Dict[str, ConversationFlow]):
        """Process a single JSON log entry"""
        timestamp = datetime.fromisoformat(log_entry.get('timestamp', ''))
        
        # Check if this is an agent communication entry
        if ('agent_name' in log_entry and 
            'conversation_id' in log_entry and 
            'operation' in log_entry):
            
            conversation_id = log_entry['conversation_id']
            
            # Initialize conversation if not exists
            if conversation_id not in conversations:
                conversations[conversation_id] = ConversationFlow(
                    conversation_id=conversation_id,
                    start_time=timestamp,
                    end_time=None,
                    total_turns=0,
                    agent_calls=[],
                    user_messages=[],
                    supervisor_responses=[]
                )
            
            flow = conversations[conversation_id]
            
            # Parse agent communication
            if log_entry['operation'].startswith('agent_communication_'):
                direction = log_entry['operation'].split('_')[-1]  # to_agent or from_agent
                
                details = log_entry.get('details', {})
                message_preview = details.get('message_preview', log_entry.get('message', ''))
                message_length = details.get('message_length', len(message_preview))
                
                agent_call = ParsedAgentCall(
                    timestamp=timestamp,
                    conversation_id=conversation_id,
                    agent_name=log_entry['agent_name'],
                    direction=direction,
                    message_content=message_preview,
                    message_length=message_length,
                    status=log_entry.get('status', 'SUCCESS')
                )
                
                flow.agent_calls.append(agent_call)
        
        # Check if this is a conversation event
        elif ('conversation_id' in log_entry and 
              'operation' in log_entry and 
              log_entry['operation'].startswith('conversation_')):
            
            conversation_id = log_entry['conversation_id']
            event_type = log_entry['operation'].split('_')[-1]  # started, ended, message
            
            if conversation_id not in conversations:
                conversations[conversation_id] = ConversationFlow(
                    conversation_id=conversation_id,
                    start_time=timestamp,
                    end_time=None,
                    total_turns=0,
                    agent_calls=[],
                    user_messages=[],
                    supervisor_responses=[]
                )
            
            flow = conversations[conversation_id]
            
            if event_type == "started":
                flow.start_time = timestamp
            elif event_type == "ended":
                flow.end_time = timestamp
                flow.conversation_ended = True
            elif event_type == "message":
                details = log_entry.get('details', {})
                direction = details.get('direction', 'unknown')
                
                if direction == "user_to_supervisor":
                    flow.user_messages.append({
                        'timestamp': timestamp.isoformat(),
                        'content': details.get('message', ''),
                        'length': details.get('message_length', 0)
                    })
                elif direction == "supervisor_to_user":
                    flow.supervisor_responses.append({
                        'timestamp': timestamp.isoformat(),
                        'content': details.get('response', ''),
                        'length': details.get('response_length', 0)
                    })
    
    def _process_text_log_entry(self, line: str, conversations: Dict[str, ConversationFlow]):
        """Process a single text log entry from agent communication logs"""
        # Parse timestamp and message using regex
        timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]'
        agent_comm_pattern = r'(🎯|📨)\s*(ORCHESTRATOR\s*→\s*(\w+)|(\w+)\s*→\s*ORCHESTRATOR):\s*(.*)'
        
        timestamp_match = re.search(timestamp_pattern, line)
        if not timestamp_match:
            return
        
        timestamp_str = timestamp_match.group(1)
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        
        agent_match = re.search(agent_comm_pattern, line)
        if not agent_match:
            return
        
        # Extract direction and agent name
        direction_emoji = agent_match.group(1)
        if direction_emoji == "🎯":  # ORCHESTRATOR → agent
            direction = "to_agent"
            agent_name = agent_match.group(3)
        else:  # agent → ORCHESTRATOR
            direction = "from_agent"
            agent_name = agent_match.group(4)
        
        message_content = agent_match.group(5)
        
        # For text logs, we need to infer conversation_id from context
        # This is a limitation - JSON logs are preferred for this reason
        # For now, we'll use a placeholder
        conversation_id = "text_log_conversation"
        
        if conversation_id not in conversations:
            conversations[conversation_id] = ConversationFlow(
                conversation_id=conversation_id,
                start_time=timestamp,
                end_time=None,
                total_turns=0,
                agent_calls=[],
                user_messages=[],
                supervisor_responses=[]
            )
        
        flow = conversations[conversation_id]
        
        agent_call = ParsedAgentCall(
            timestamp=timestamp,
            conversation_id=conversation_id,
            agent_name=agent_name,
            direction=direction,
            message_content=message_content,
            message_length=len(message_content),
            status="SUCCESS"
        )
        
        flow.agent_calls.append(agent_call)
    
    def _finalize_conversation_flow(self, flow: ConversationFlow):
        """Finalize conversation flow calculations"""
        # Calculate duration
        if flow.end_time and flow.start_time:
            flow.duration_seconds = (flow.end_time - flow.start_time).total_seconds()
        
        # Estimate total turns (approximate from messages)
        flow.total_turns = max(len(flow.user_messages), len(flow.supervisor_responses))
        
        # Add turn numbers to agent calls (estimate based on timestamp order)
        if flow.agent_calls:
            # Sort by timestamp
            flow.agent_calls.sort(key=lambda x: x.timestamp)
            
            # Estimate turn numbers
            current_turn = 1
            last_timestamp = None
            
            for call in flow.agent_calls:
                if last_timestamp and (call.timestamp - last_timestamp).total_seconds() > 5:
                    current_turn += 1
                call.turn_number = current_turn
                last_timestamp = call.timestamp

class FlowComparator:
    """Compares actual flows against expected scenario flows"""
    
    def __init__(self):
        self.logger = logging.getLogger('flow_comparator')
    
    def compare_with_scenario(self, actual_flow: ConversationFlow, expected_scenario: Any) -> Dict[str, Any]:
        """Compare actual conversation flow with expected scenario"""
        comparison = {
            "conversation_id": actual_flow.conversation_id,
            "scenario_id": expected_scenario.scenario_id if hasattr(expected_scenario, 'scenario_id') else 'unknown',
            "matches": [],
            "discrepancies": [],
            "score": 0.0,
            "success": False
        }
        
        # Check agent usage
        expected_agents = self._extract_expected_agents_from_scenario(expected_scenario)
        actual_agents = actual_flow.get_unique_agents_used()
        
        # Agent usage comparison
        for agent in expected_agents:
            if agent in actual_agents:
                comparison["matches"].append(f"Agent {agent} was used as expected")
            else:
                comparison["discrepancies"].append(f"Agent {agent} was expected but not used")
        
        for agent in actual_agents:
            if agent not in expected_agents:
                comparison["discrepancies"].append(f"Agent {agent} was used but not expected")
        
        # Turn count comparison (if scenario has turn expectations)
        if hasattr(expected_scenario, 'estimated_turns'):
            min_turns, max_turns = expected_scenario.estimated_turns
            if min_turns <= actual_flow.total_turns <= max_turns:
                comparison["matches"].append(f"Turn count {actual_flow.total_turns} within expected range {min_turns}-{max_turns}")
            else:
                comparison["discrepancies"].append(f"Turn count {actual_flow.total_turns} outside expected range {min_turns}-{max_turns}")
        
        # Calculate score
        total_checks = len(comparison["matches"]) + len(comparison["discrepancies"])
        if total_checks > 0:
            comparison["score"] = len(comparison["matches"]) / total_checks
            comparison["success"] = comparison["score"] >= 0.8
        
        return comparison
    
    def _extract_expected_agents_from_scenario(self, scenario) -> List[str]:
        """Extract expected agent names from scenario definition"""
        if not hasattr(scenario, 'success_criteria'):
            return []
        
        criteria = scenario.success_criteria
        expected_agents = []
        
        if 'agent_calls' in criteria:
            agent_calls = criteria['agent_calls']
            if isinstance(agent_calls, list):
                expected_agents = agent_calls
        
        return expected_agents

# Convenience functions
def parse_conversation_logs(log_dir: str = "logs") -> List[ConversationFlow]:
    """Parse all available logs and return conversation flows"""
    parser = LogParser(log_dir)
    
    # Try JSON logs first (more structured)
    flows = parser.parse_json_logs()
    
    # If no JSON logs, try agent communication logs
    if not flows:
        flows = parser.parse_agent_communication_logs()
    
    return flows

def extract_flow_for_conversation(conversation_id: str, log_dir: str = "logs") -> Optional[ConversationFlow]:
    """Extract flow for a specific conversation ID"""
    flows = parse_conversation_logs(log_dir)
    
    for flow in flows:
        if flow.conversation_id == conversation_id:
            return flow
    
    return None

def analyze_recent_conversations(log_dir: str = "logs", hours: int = 24) -> List[ConversationFlow]:
    """Analyze conversations from the last N hours"""
    from datetime import timedelta
    flows = parse_conversation_logs(log_dir)
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    recent_flows = [flow for flow in flows if flow.start_time >= cutoff_time]
    
    return recent_flows 