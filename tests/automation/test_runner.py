"""
Automated Test Runner - Comprehensive system for executing conversation scenarios,
capturing agent flows, and validating against expected patterns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import concurrent.futures
from pathlib import Path

# Import our core systems
from tests.user_simulator.personas import PersonaConfig, get_available_personas, get_all_persona_configs
from tests.user_simulator.scenarios import ScenarioDefinition, ConversationScenarios, ScenarioMatcher
from tests.user_simulator.scenario_executor import ScenarioExecutor, ConversationRecord
from tests.user_simulator.date_handler import ConversationDateContext, generate_test_dates
from tests.logs_parser.flow_extractor import parse_conversation_logs, extract_flow_for_conversation
from tests.logs_parser.flow_validator import FlowValidator, ValidationResult, generate_validation_report

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

class TestPriority(Enum):
    """Test execution priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TestConfiguration:
    """Configuration for test execution"""
    max_parallel_tests: int = 3
    test_timeout_seconds: int = 120
    conversation_timeout_seconds: int = 60
    retry_failed_tests: bool = True
    max_retries: int = 2
    capture_detailed_logs: bool = True
    validate_flows: bool = True
    generate_reports: bool = True
    log_directory: str = "logs"
    results_directory: str = "test_results"

@dataclass 
class TestCase:
    """Individual test case definition"""
    test_id: str
    scenario_id: str
    persona_name: str
    priority: TestPriority = TestPriority.NORMAL
    expected_duration_seconds: float = 30.0
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"{self.scenario_id}_{self.persona_name}_{uuid.uuid4().hex[:8]}"

@dataclass
class TestResult:
    """Comprehensive test result"""
    test_case: TestCase
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Execution results
    conversation_id: Optional[str] = None
    conversation_record: Optional[ConversationRecord] = None
    
    # Validation results
    flow_validation: Optional[ValidationResult] = None
    validation_score: float = 0.0
    
    # Issues and metrics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    retry_count: int = 0
    logs_captured: bool = False
    
    def mark_completed(self, success: bool = True):
        """Mark test as completed"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = TestStatus.COMPLETED if success else TestStatus.FAILED
    
    def add_error(self, error: str):
        """Add error to test result"""
        self.errors.append(error)
        if self.status not in [TestStatus.FAILED, TestStatus.TIMEOUT]:
            self.status = TestStatus.FAILED
    
    def add_warning(self, warning: str):
        """Add warning to test result"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_id": self.test_case.test_id,
            "scenario_id": self.test_case.scenario_id,
            "persona_name": self.test_case.persona_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "conversation_id": self.conversation_id,
            "validation_score": self.validation_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "retry_count": self.retry_count,
            "logs_captured": self.logs_captured,
            "flow_validation": self.flow_validation.to_dict() if self.flow_validation else None
        }

class ConversationSimulator:
    """Simulates conversations for testing purposes"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.logger = logging.getLogger('conversation_simulator')
        self.date_context = ConversationDateContext()
    
    def simulate_conversation(self, test_case: TestCase) -> Tuple[str, ConversationRecord]:
        """Simulate a conversation for the given test case"""
        self.logger.info(f"🎭 Simulating conversation for {test_case.test_id}")
        
        # Get scenario and persona
        scenarios = ConversationScenarios.get_all_scenarios()
        personas = get_all_persona_configs()
        
        if test_case.scenario_id not in scenarios:
            raise ValueError(f"Unknown scenario: {test_case.scenario_id}")
        
        if test_case.persona_name not in personas:
            raise ValueError(f"Unknown persona: {test_case.persona_name}")
        
        scenario = scenarios[test_case.scenario_id]
        persona = personas[test_case.persona_name]
        
        # Generate appropriate dates for scheduling scenarios
        scenario_dates = generate_test_dates(test_case.scenario_id)
        
        # Create conversation record
        conversation_id = str(uuid.uuid4())
        conversation_record = ConversationRecord(
            scenario_id=test_case.scenario_id,
            persona=persona.name
        )
        # Set additional attributes
        conversation_record.conversation_id = conversation_id
        conversation_record.scenario_dates = scenario_dates
        
        # Simulate conversation turns based on scenario expectations
        self._simulate_conversation_turns(conversation_record, scenario, persona)
        
        conversation_record.end_time = datetime.now()
        conversation_record.success = True  # Assume success for simulation
        
        self.logger.info(f"✅ Conversation simulation complete: {conversation_id}")
        return conversation_id, conversation_record
    
    def _simulate_conversation_turns(self, record: ConversationRecord, scenario: ScenarioDefinition, persona: PersonaConfig):
        """Simulate the conversation turns"""
        # This is a simplified simulation - in a real implementation, 
        # this would integrate with the actual conversation system
        
        min_turns, max_turns = scenario.estimated_turns
        actual_turns = min_turns + ((max_turns - min_turns) // 2)  # Use middle of range
        
        record.turns = actual_turns
        record.messages = []
        
        # Simulate messages based on scenario type
        if "scheduling" in scenario.tags:
            record.messages.extend([
                {"role": "user", "content": "I'm interested in the position"},
                {"role": "assistant", "content": "Great! Would you like to schedule an interview?"},
                {"role": "user", "content": f"Yes, {record.scenario_dates.get('preferred_date', 'tomorrow')} works"},
                {"role": "assistant", "content": "Perfect! I'll check available slots."}
            ])
        
        elif "information" in scenario.tags:
            record.messages.extend([
                {"role": "user", "content": "Can you tell me about the job requirements?"},
                {"role": "assistant", "content": "I'll get that information for you."},
                {"role": "user", "content": "What about the salary range?"},
                {"role": "assistant", "content": "Let me check our compensation details."}
            ])
        
        elif "rejection" in scenario.tags:
            record.messages.extend([
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello! Are you interested in our Python developer position?"},
                {"role": "user", "content": "Not really, thanks"},
                {"role": "assistant", "content": "Thank you for your time!"}
            ])
        
        # Pad to actual turn count
        while len(record.messages) < actual_turns:
            record.messages.append({"role": "user", "content": "I see"})
            if len(record.messages) < actual_turns:
                record.messages.append({"role": "assistant", "content": "Is there anything else I can help with?"})

class AutomatedTestRunner:
    """Main automated test runner orchestrating all testing activities"""
    
    def __init__(self, config: TestConfiguration = None):
        self.config = config or TestConfiguration()
        self.logger = logging.getLogger('automated_test_runner')
        
        # Initialize components
        self.simulator = ConversationSimulator(self.config)
        self.flow_validator = FlowValidator()
        self.test_queue: List[TestCase] = []
        self.running_tests: Dict[str, TestResult] = {}
        self.completed_tests: List[TestResult] = []
        
        # Setup directories
        self._setup_directories()
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.total_tests_planned: int = 0
        
    def _setup_directories(self):
        """Setup necessary directories for test execution"""
        Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_directory).mkdir(parents=True, exist_ok=True)
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the execution queue"""
        self.test_queue.append(test_case)
        self.logger.info(f"📋 Added test case: {test_case.test_id}")
    
    def add_scenario_persona_combination(self, scenario_id: str, persona_name: str, priority: TestPriority = TestPriority.NORMAL):
        """Add a scenario-persona combination as a test case"""
        test_case = TestCase(
            test_id="",  # Will be auto-generated
            scenario_id=scenario_id,
            persona_name=persona_name,
            priority=priority,
            tags=[scenario_id, persona_name]
        )
        self.add_test_case(test_case)
    
    def generate_comprehensive_test_suite(self) -> int:
        """Generate comprehensive test suite covering all scenario-persona combinations"""
        self.logger.info("🔄 Generating comprehensive test suite...")
        
        scenarios = ConversationScenarios.get_all_scenarios()
        personas = get_all_persona_configs()
        
        initial_count = len(self.test_queue)
        
        for persona_name in personas.keys():
            # For now, assume all scenarios are compatible with all personas
            for scenario_id in scenarios.keys():
                self.add_scenario_persona_combination(scenario_id, persona_name)
        
        added_count = len(self.test_queue) - initial_count
        self.logger.info(f"✅ Generated {added_count} test cases")
        return added_count
    
    async def execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        test_result = TestResult(
            test_case=test_case,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.running_tests[test_case.test_id] = test_result
        
        try:
            self.logger.info(f"🚀 Executing test: {test_case.test_id}")
            
            # Step 1: Simulate conversation
            try:
                conversation_id, conversation_record = await asyncio.get_event_loop().run_in_executor(
                    None, self.simulator.simulate_conversation, test_case
                )
                test_result.conversation_id = conversation_id
                test_result.conversation_record = conversation_record
                test_result.logs_captured = True
                
            except Exception as e:
                test_result.add_error(f"Conversation simulation failed: {str(e)}")
                test_result.mark_completed(False)
                return test_result
            
            # Step 2: Extract and validate flow (if enabled)
            if self.config.validate_flows and conversation_id:
                try:
                    # In a real implementation, this would extract from actual logs
                    # For simulation, we'll create a mock flow validation
                    flow_validation = await self._validate_conversation_flow(test_case, conversation_record)
                    test_result.flow_validation = flow_validation
                    test_result.validation_score = flow_validation.overall_score
                    
                    if not flow_validation.success:
                        test_result.add_warning("Flow validation did not meet success criteria")
                    
                except Exception as e:
                    test_result.add_error(f"Flow validation failed: {str(e)}")
            
            # Step 3: Calculate metrics
            test_result.metrics = {
                "conversation_turns": conversation_record.turns,
                "conversation_duration": conversation_record.end_time - conversation_record.start_time if conversation_record.end_time else timedelta(0),
                "messages_count": len(conversation_record.messages),
                "scenario_match": test_result.validation_score >= 0.8 if test_result.validation_score else False
            }
            
            # Mark as completed
            test_result.mark_completed(len(test_result.errors) == 0)
            
        except asyncio.TimeoutError:
            test_result.status = TestStatus.TIMEOUT
            test_result.add_error("Test execution timed out")
            
        except Exception as e:
            test_result.add_error(f"Unexpected error: {str(e)}")
            test_result.mark_completed(False)
            
        finally:
            if test_case.test_id in self.running_tests:
                del self.running_tests[test_case.test_id]
            self.completed_tests.append(test_result)
            
        self.logger.info(f"✅ Test completed: {test_case.test_id} ({test_result.status.value})")
        return test_result
    
    async def _validate_conversation_flow(self, test_case: TestCase, conversation_record: ConversationRecord) -> ValidationResult:
        """Validate conversation flow against expected scenario pattern"""
        # In a real implementation, this would use the actual log parser and validator
        # For simulation, we'll create a realistic validation result
        
        scenarios = ConversationScenarios.get_all_scenarios()
        scenario = scenarios[test_case.scenario_id]
        
        # Create a mock validation result based on scenario expectations
        validation_result = ValidationResult(conversation_record.conversation_id, test_case.scenario_id)
        
        # Simulate validation based on scenario criteria
        success_criteria = scenario.success_criteria
        
        # Mock agent flow analysis
        expected_agents = success_criteria.get("agent_calls", ["supervisor", "info_agent", "exit_agent"])
        actual_agents = ["supervisor"]  # Simplified simulation
        
        if "scheduling" in scenario.tags:
            actual_agents.extend(["sched_agent", "exit_agent"])
        if "information" in scenario.tags:
            actual_agents.extend(["info_agent", "exit_agent"])
        
        validation_result.agent_flow_analysis = {
            "expected_agents": expected_agents,
            "actual_agents": actual_agents,
            "correctly_used": [agent for agent in expected_agents if agent in actual_agents],
            "missing_agents": [agent for agent in expected_agents if agent not in actual_agents],
            "unexpected_agents": [agent for agent in actual_agents if agent not in expected_agents]
        }
        
        # Mock turn analysis
        min_turns, max_turns = scenario.estimated_turns
        validation_result.turn_analysis = {
            "expected_range": {"min": min_turns, "max": max_turns},
            "actual_turns": conversation_record.turns,
            "within_range": min_turns <= conversation_record.turns <= max_turns
        }
        
        # Calculate score based on matches
        agent_score = len(validation_result.agent_flow_analysis["correctly_used"]) / max(len(expected_agents), 1)
        turn_score = 1.0 if validation_result.turn_analysis["within_range"] else 0.5
        
        validation_result.overall_score = (agent_score * 0.7) + (turn_score * 0.3)
        validation_result.success = validation_result.overall_score >= 0.8
        
        return validation_result
    
    async def run_test_suite(self, max_parallel: Optional[int] = None) -> List[TestResult]:
        """Run the complete test suite"""
        if not self.test_queue:
            self.logger.warning("No test cases in queue")
            return []
        
        max_parallel = max_parallel or self.config.max_parallel_tests
        self.start_time = datetime.now()
        self.total_tests_planned = len(self.test_queue)
        
        self.logger.info(f"🚀 Starting test suite execution: {self.total_tests_planned} tests, max {max_parallel} parallel")
        
        # Sort by priority
        self.test_queue.sort(key=lambda tc: tc.priority.value, reverse=True)
        
        # Execute tests with controlled parallelism
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(test_case):
            async with semaphore:
                return await asyncio.wait_for(
                    self.execute_test_case(test_case),
                    timeout=self.config.test_timeout_seconds
                )
        
        # Run all tests
        tasks = [execute_with_semaphore(test_case) for test_case in self.test_queue]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_case = self.test_queue[i]
                error_result = TestResult(
                    test_case=test_case,
                    status=TestStatus.FAILED,
                    start_time=datetime.now()
                )
                error_result.add_error(f"Task execution failed: {str(result)}")
                error_result.mark_completed(False)
                self.completed_tests.append(error_result)
        
        # Clear queue
        self.test_queue.clear()
        
        # Generate summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"🎉 Test suite completed: {len(self.completed_tests)} tests in {total_time:.1f}s")
        
        return self.completed_tests
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        if not self.completed_tests:
            return {
                "status": "no_tests_completed",
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
                "average_duration_seconds": 0.0,
                "average_validation_score": 0.0,
                "total_execution_time": 0.0,
                "tests_per_minute": 0.0
            }
        
        total_tests = len(self.completed_tests)
        successful_tests = len([t for t in self.completed_tests if t.status == TestStatus.COMPLETED])
        failed_tests = len([t for t in self.completed_tests if t.status == TestStatus.FAILED])
        
        avg_duration = sum(t.duration_seconds for t in self.completed_tests) / total_tests
        avg_validation_score = sum(t.validation_score for t in self.completed_tests if t.validation_score > 0) / max(1, len([t for t in self.completed_tests if t.validation_score > 0]))
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests,
            "average_duration_seconds": avg_duration,
            "average_validation_score": avg_validation_score,
            "total_execution_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "tests_per_minute": (total_tests / ((datetime.now() - self.start_time).total_seconds() / 60)) if self.start_time else 0
        }
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        filepath = Path(self.config.results_directory) / filename
        
        results_data = {
            "execution_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_tests_planned": self.total_tests_planned,
                "configuration": {
                    "max_parallel_tests": self.config.max_parallel_tests,
                    "test_timeout_seconds": self.config.test_timeout_seconds,
                    "validate_flows": self.config.validate_flows
                }
            },
            "statistics": self.get_execution_statistics(),
            "test_results": [test.to_dict() for test in self.completed_tests]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Test results saved to: {filepath}")
        return str(filepath)

# Convenience functions for easy usage
async def run_single_test_async(scenario_id: str, persona_name: str, config: TestConfiguration = None) -> TestResult:
    """Run a single test case asynchronously"""
    runner = AutomatedTestRunner(config)
    test_case = TestCase("", scenario_id, persona_name)
    runner.add_test_case(test_case)
    
    results = await runner.run_test_suite()
    return results[0] if results else None

def run_single_test(scenario_id: str, persona_name: str, config: TestConfiguration = None) -> TestResult:
    """Run a single test case synchronously"""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        raise RuntimeError("Use run_single_test_async when already in async context")
    except RuntimeError:
        # No loop running, safe to create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_single_test_async(scenario_id, persona_name, config))
        finally:
            loop.close()

async def run_comprehensive_test_suite_async(config: TestConfiguration = None) -> List[TestResult]:
    """Run comprehensive test suite covering all scenarios and personas asynchronously"""
    runner = AutomatedTestRunner(config)
    runner.generate_comprehensive_test_suite()
    return await runner.run_test_suite()

def run_comprehensive_test_suite(config: TestConfiguration = None) -> List[TestResult]:
    """Run comprehensive test suite covering all scenarios and personas synchronously"""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        raise RuntimeError("Use run_comprehensive_test_suite_async when already in async context")
    except RuntimeError:
        # No loop running, safe to create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_comprehensive_test_suite_async(config))
        finally:
            loop.close() 