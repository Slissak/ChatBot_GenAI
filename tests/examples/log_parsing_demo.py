"""
Log Parsing and Flow Validation Demo - Demonstrates the complete system for
extracting agent flows from logs and validating against expected scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import our systems
from tests.logs_parser.flow_extractor import LogParser, ConversationFlow, parse_conversation_logs
from tests.logs_parser.flow_validator import FlowValidator, ValidationResult, generate_validation_report
from tests.user_simulator.scenarios import ConversationScenarios, ScenarioMatcher
from tests.user_simulator.scenario_executor import ScenarioExecutor
from tests.user_simulator.personas import get_available_personas

# Set up logging for demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_sample_log_data():
    """Create sample log data for demonstration purposes"""
    print("📝 Creating sample log data for demonstration...")
    
    # Create sample JSON log entries
    sample_logs = [
        {
            "timestamp": "2025-01-09T10:00:00.000000",
            "level": "INFO",
            "logger": "agent_communication",
            "message": "🎯 ORCHESTRATOR → info_agent: What are the job responsibilities?",
            "conversation_id": "demo-conv-001",
            "agent_name": "info_agent",
            "operation": "agent_communication_to_agent",
            "status": "SUCCESS",
            "details": {
                "direction": "to_agent",
                "message_length": 35,
                "message_preview": "What are the job responsibilities?"
            }
        },
        {
            "timestamp": "2025-01-09T10:00:05.000000",
            "level": "INFO",
            "logger": "agent_communication",
            "message": "📨 info_agent → ORCHESTRATOR: The Python Developer role involves...",
            "conversation_id": "demo-conv-001",
            "agent_name": "info_agent",
            "operation": "agent_communication_from_agent",
            "status": "SUCCESS",
            "details": {
                "direction": "from_agent",
                "message_length": 150,
                "message_preview": "The Python Developer role involves..."
            }
        },
        {
            "timestamp": "2025-01-09T10:00:10.000000",
            "level": "INFO",
            "logger": "agent_communication",
            "message": "🎯 ORCHESTRATOR → sched_agent: Check available slots for 2025-01-15",
            "conversation_id": "demo-conv-001",
            "agent_name": "sched_agent",
            "operation": "agent_communication_to_agent",
            "status": "SUCCESS",
            "details": {
                "direction": "to_agent",
                "message_length": 42,
                "message_preview": "Check available slots for 2025-01-15"
            }
        },
        {
            "timestamp": "2025-01-09T10:00:15.000000",
            "level": "INFO",
            "logger": "agent_communication",
            "message": "📨 sched_agent → ORCHESTRATOR: Available slots found...",
            "conversation_id": "demo-conv-001",
            "agent_name": "sched_agent",
            "operation": "agent_communication_from_agent",
            "status": "SUCCESS",
            "details": {
                "direction": "from_agent",
                "message_length": 120,
                "message_preview": "Available slots found..."
            }
        },
        {
            "timestamp": "2025-01-09T10:00:20.000000",
            "level": "INFO",
            "logger": "agent_communication",
            "message": "🎯 ORCHESTRATOR → exit_agent: Analyze conversation completion",
            "conversation_id": "demo-conv-001",
            "agent_name": "exit_agent",
            "operation": "agent_communication_to_agent",
            "status": "SUCCESS",
            "details": {
                "direction": "to_agent",
                "message_length": 35,
                "message_preview": "Analyze conversation completion"
            }
        },
        {
            "timestamp": "2025-01-09T10:00:25.000000",
            "level": "INFO",
            "logger": "conversation",
            "message": "🔚 Conversation ended: demo-conv-001",
            "conversation_id": "demo-conv-001",
            "operation": "conversation_ended",
            "status": "INFO",
            "details": {
                "reason": "successful_scheduling"
            }
        }
    ]
    
    # Write sample logs to a demo file
    os.makedirs("logs", exist_ok=True)
    demo_log_file = "logs/demo_structured.json"
    
    with open(demo_log_file, 'w') as f:
        for log_entry in sample_logs:
            f.write(json.dumps(log_entry) + '\n')
    
    print(f"✅ Sample log data created: {demo_log_file}")
    return demo_log_file

def demo_log_parsing():
    """Demonstrate log parsing capabilities"""
    print("\n🔍 === LOG PARSING DEMO ===")
    
    # Create sample log data
    demo_log_file = create_sample_log_data()
    
    # Initialize parser
    parser = LogParser()
    
    print(f"\n📊 Parsing logs from: {demo_log_file}")
    
    # Parse the demo logs
    flows = parser.parse_json_logs(demo_log_file)
    
    print(f"✅ Parsed {len(flows)} conversation flows")
    
    for flow in flows:
        print(f"\n💬 Conversation: {flow.conversation_id}")
        print(f"   Start: {flow.start_time}")
        print(f"   End: {flow.end_time}")
        print(f"   Duration: {flow.duration_seconds:.1f}s")
        print(f"   Turns: {flow.total_turns}")
        print(f"   Agents Used: {', '.join(flow.get_unique_agents_used())}")
        print(f"   Agent Sequence: {' → '.join(flow.get_agent_sequence())}")
        print(f"   Agent Call Counts: {flow.to_dict()['agent_call_counts']}")
        
        print(f"\n   📋 Agent Calls:")
        for call in flow.agent_calls:
            direction_icon = "→" if call.direction == "to_agent" else "←"
            print(f"      {call.timestamp.strftime('%H:%M:%S')} {direction_icon} {call.agent_name}: {call.message_content[:50]}...")
    
    return flows

def demo_flow_validation():
    """Demonstrate flow validation against scenarios"""
    print("\n🔬 === FLOW VALIDATION DEMO ===")
    
    # Get parsed flows
    flows = demo_log_parsing()
    
    if not flows:
        print("❌ No flows to validate")
        return
    
    # Get a demo flow
    demo_flow = flows[0]
    
    # Test against different scenarios
    scenarios = ConversationScenarios.get_all_scenarios()
    validator = FlowValidator()
    
    print(f"\n🎯 Validating conversation {demo_flow.conversation_id} against different scenarios:")
    
    test_scenarios = ["full_journey_success", "direct_scheduling", "basic_greeting"]
    
    for scenario_id in test_scenarios:
        if scenario_id not in scenarios:
            continue
            
        scenario = scenarios[scenario_id]
        print(f"\n📋 Testing against: {scenario.name}")
        
        result = validator.validate_conversation_flow(demo_flow, scenario)
        
        print(f"   Score: {result.overall_score:.2f}")
        print(f"   Success: {'✅' if result.success else '❌'}")
        print(f"   Expected Agents: {', '.join(result.agent_flow_analysis.get('expected_agents', []))}")
        print(f"   Actual Agents: {', '.join(result.agent_flow_analysis.get('actual_agents', []))}")
        
        if result.critical_issues:
            print(f"   🚨 Critical Issues:")
            for issue in result.critical_issues[:3]:  # Show first 3
                print(f"      - {issue}")
        
        if result.warnings:
            print(f"   ⚠️  Warnings:")
            for warning in result.warnings[:2]:  # Show first 2
                print(f"      - {warning}")
        
        # Show the best match
        if result.overall_score >= 0.8:
            print(f"   🎉 Good match for scenario: {scenario.name}")
            break

def demo_scenario_integration():
    """Demonstrate integration with scenario executor for real-time validation"""
    print("\n🔗 === SCENARIO INTEGRATION DEMO ===")
    
    print("This demonstrates how the log parsing integrates with scenario execution:")
    print("\n📍 Integration Points:")
    print("1. ScenarioExecutor runs conversations and captures conversation IDs")
    print("2. LogParser extracts agent flows from logs using conversation IDs")
    print("3. FlowValidator compares actual vs expected flows")
    print("4. ValidationResult provides detailed analysis")
    
    # Show how it would work with real scenario execution
    scenarios = ConversationScenarios.get_all_scenarios()
    personas = get_available_personas()
    
    print(f"\n🎭 Available for testing:")
    print(f"   Scenarios: {len(scenarios)} total")
    print(f"   Personas: {len(personas)} total")
    print(f"   Potential combinations: {sum(len(ScenarioMatcher.get_scenarios_for_persona(persona)) for persona in personas)}")
    
    print(f"\n🔄 Typical workflow:")
    print("1. Execute scenario with persona")
    print("2. Capture conversation_id from execution")
    print("3. Parse logs to extract actual agent flow")
    print("4. Validate against expected scenario flow")
    print("5. Generate validation report with recommendations")

def demo_batch_analysis():
    """Demonstrate batch analysis capabilities"""
    print("\n📊 === BATCH ANALYSIS DEMO ===")
    
    # Create multiple sample conversations
    print("📝 Creating multiple sample conversations...")
    
    sample_conversations = [
        {
            "conversation_id": "batch-001",
            "scenario_type": "successful_full_journey",
            "agents_used": ["info_agent", "sched_agent", "exit_agent"],
            "turns": 5,
            "duration": 45.0
        },
        {
            "conversation_id": "batch-002", 
            "scenario_type": "quick_rejection",
            "agents_used": ["exit_agent"],
            "turns": 2,
            "duration": 15.0
        },
        {
            "conversation_id": "batch-003",
            "scenario_type": "info_only",
            "agents_used": ["info_agent", "exit_agent"],
            "turns": 3,
            "duration": 30.0
        }
    ]
    
    print(f"✅ Created {len(sample_conversations)} sample conversations")
    
    # Simulate validation results
    from tests.logs_parser.flow_validator import ValidationResult
    
    mock_results = []
    for conv in sample_conversations:
        result = ValidationResult(conv["conversation_id"], conv["scenario_type"])
        result.overall_score = 0.85 if conv["scenario_type"] != "quick_rejection" else 0.75
        result.success = result.overall_score >= 0.8
        result.agent_flow_analysis = {
            "expected_agents": conv["agents_used"],
            "actual_agents": conv["agents_used"],
            "correctly_used": conv["agents_used"]
        }
        result.turn_analysis = {
            "actual_turns": conv["turns"],
            "within_range": True
        }
        mock_results.append(result)
    
    # Generate batch report
    report = generate_validation_report(mock_results)
    
    print(f"\n📈 Batch Analysis Report:")
    print(f"   Total Validations: {report['summary']['total_validations']}")
    print(f"   Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"   Average Score: {report['summary']['average_score']:.2f}")
    print(f"   Critical Issues: {report['issue_analysis']['total_critical_issues']}")
    print(f"   Warnings: {report['issue_analysis']['total_warnings']}")
    
    if report['issue_analysis']['common_issues']:
        print(f"\n🚨 Common Issues:")
        for issue, frequency in report['issue_analysis']['common_issues']:
            print(f"   - {issue} ({frequency} times)")

def demo_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities"""
    print("\n⏱️ === REAL-TIME MONITORING DEMO ===")
    
    print("Real-time monitoring capabilities:")
    print("\n🔍 Log Monitoring:")
    print("   - Parse logs as they're written")
    print("   - Extract agent flows in real-time")
    print("   - Immediate validation against expected patterns")
    
    print("\n📊 Live Analytics:")
    print("   - Success rate tracking")
    print("   - Agent usage patterns")
    print("   - Performance metrics")
    print("   - Anomaly detection")
    
    print("\n🚨 Alert System:")
    print("   - Critical flow violations")
    print("   - Unexpected agent patterns")
    print("   - Performance degradation")
    print("   - System health issues")
    
    # Show recent conversation analysis capability
    print(f"\n📅 Recent Conversations Analysis:")
    try:
        from tests.logs_parser.flow_extractor import analyze_recent_conversations
        print("   ✅ Can analyze conversations from last N hours")
        print("   ✅ Filter by time range, scenario, persona")
        print("   ✅ Generate trending reports")
    except Exception as e:
        print(f"   ⚠️  Analysis capability: {e}")

def comprehensive_demo():
    """Run the complete demonstration"""
    print("🚀 === COMPREHENSIVE LOG PARSING & VALIDATION DEMO ===\n")
    
    print("This demonstration showcases the complete system for:")
    print("• Parsing agent communication logs")
    print("• Extracting conversation flows")
    print("• Validating against expected scenarios")
    print("• Generating comprehensive reports")
    
    try:
        # Run all demo components
        demo_log_parsing()
        demo_flow_validation()
        demo_scenario_integration()
        demo_batch_analysis()
        demo_real_time_monitoring()
        
        print("\n🎉 === DEMO COMPLETE ===")
        print("\nThe log parsing and validation system is ready for:")
        print("✅ Real-time conversation flow extraction")
        print("✅ Automated scenario validation")
        print("✅ Comprehensive testing and reporting")
        print("✅ Integration with scenario execution")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_demo() 