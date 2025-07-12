"""
Scenario Execution Demo - Demonstrates how conversation scenarios work
with the multi-agent system and different persona behaviors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.user_simulator.scenarios import ConversationScenarios, ScenarioMatcher
from tests.user_simulator.scenario_executor import ScenarioExecutor, quick_scenario_demo
from tests.user_simulator.personas import get_available_personas
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def show_all_scenarios():
    """Display all available scenarios with their details"""
    print("🎬 === ALL AVAILABLE SCENARIOS ===\n")
    
    scenarios = ConversationScenarios.get_all_scenarios()
    
    for scenario_id, scenario in scenarios.items():
        print(f"📋 **{scenario.name}** ({scenario_id})")
        print(f"   Primary Persona: {scenario.primary_persona}")
        print(f"   Compatible Personas: {', '.join(scenario.compatible_personas)}")
        print(f"   Expected Turns: {scenario.estimated_turns[0]}-{scenario.estimated_turns[1]}")
        print(f"   Description: {scenario.description}")
        print(f"   Tags: {', '.join(scenario.tags)}")
        print(f"   Success Criteria: {list(scenario.success_criteria.keys())}")
        print()

def show_persona_scenario_mapping():
    """Show which scenarios each persona can participate in"""
    print("🎭 === PERSONA-SCENARIO MAPPING ===\n")
    
    personas = get_available_personas()
    
    for persona in personas:
        scenarios = ScenarioMatcher.get_scenarios_for_persona(persona)
        primary_scenario = ScenarioMatcher.get_primary_scenario_for_persona(persona)
        
        print(f"👤 **{persona.upper()}**")
        if primary_scenario:
            print(f"   Primary Scenario: {primary_scenario.name}")
        print(f"   Compatible Scenarios ({len(scenarios)}):")
        for scenario in scenarios:
            indicator = "🎯" if scenario.primary_persona == persona else "✓"
            print(f"      {indicator} {scenario.name}")
        print()

def demo_single_scenario():
    """Interactive demo of a single scenario"""
    print("🎬 === SINGLE SCENARIO DEMO ===\n")
    
    scenarios = ConversationScenarios.get_all_scenarios()
    scenario_list = list(scenarios.keys())
    
    print("Available scenarios:")
    for i, scenario_id in enumerate(scenario_list, 1):
        scenario = scenarios[scenario_id]
        print(f"{i}. {scenario.name} (Primary: {scenario.primary_persona})")
    
    try:
        choice = int(input("\nSelect scenario number: ")) - 1
        if choice < 0 or choice >= len(scenario_list):
            print("Invalid choice!")
            return
        
        scenario_id = scenario_list[choice]
        scenario = scenarios[scenario_id]
        
        print(f"\n🎭 Selected: {scenario.name}")
        print(f"Primary persona: {scenario.primary_persona}")
        print(f"Compatible personas: {', '.join(scenario.compatible_personas)}")
        
        # Ask for persona choice
        print(f"\nAvailable personas for this scenario:")
        all_personas = [scenario.primary_persona] + [p for p in scenario.compatible_personas if p != scenario.primary_persona]
        for i, persona in enumerate(all_personas, 1):
            print(f"{i}. {persona}")
        
        persona_choice = int(input("\nSelect persona number (or press Enter for primary): ") or "1") - 1
        if persona_choice < 0 or persona_choice >= len(all_personas):
            print("Invalid choice, using primary persona")
            persona_choice = 0
        
        selected_persona = all_personas[persona_choice]
        
        print(f"\n🚀 Running scenario '{scenario.name}' with persona '{selected_persona}'...")
        print("=" * 60)
        
        # Execute scenario
        record, validation = quick_scenario_demo(scenario_id)
        
        print("\n📊 === EXECUTION RESULTS ===")
        print(f"Scenario: {scenario.name}")
        print(f"Persona: {selected_persona}")
        print(f"Conversation ID: {record.conversation_id}")
        print(f"Total Turns: {record.turns}")
        print(f"Duration: {record.get_duration():.1f} seconds")
        print(f"Conversation Ended: {record.conversation_ended}")
        print(f"Success: {record.success}")
        print(f"Validation Score: {validation['score']:.2%}")
        
        if record.errors:
            print(f"Errors: {', '.join(record.errors)}")
        
        # Show message flow
        print(f"\n💬 === MESSAGE FLOW ===")
        for i, message in enumerate(record.messages, 1):
            role_icon = "👤" if message['role'] == 'user' else "🤖"
            content = message['content'][:100] + "..." if len(message['content']) > 100 else message['content']
            print(f"{i}. {role_icon} {message['role'].title()}: {content}")
        
    except ValueError:
        print("Please enter a valid number!")
    except KeyboardInterrupt:
        print("\nDemo interrupted!")

def demo_persona_scenario_combinations():
    """Demo showing how different personas behave in the same scenario"""
    print("🎭 === PERSONA COMPARISON DEMO ===\n")
    
    # Use a scenario that works with multiple personas
    scenario_id = "info_query_interested"
    scenarios = ConversationScenarios.get_all_scenarios()
    scenario = scenarios[scenario_id]
    
    print(f"🎬 Testing scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    
    executor = ScenarioExecutor()
    results = []
    
    # Test with primary persona and a few compatible ones
    test_personas = [scenario.primary_persona] + scenario.compatible_personas[:2]
    
    for persona in test_personas:
        print(f"\n👤 Testing with {persona} persona...")
        try:
            record = executor.execute_scenario(scenario_id, persona, max_turns=6)
            validation = executor.validate_scenario_execution(record)
            results.append((persona, record, validation))
            
            print(f"   ✅ Completed - Turns: {record.turns}, Success: {record.success}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Compare results
    print(f"\n📊 === COMPARISON RESULTS ===")
    print(f"{'Persona':<15} {'Turns':<6} {'Duration':<10} {'Success':<8} {'Score':<8}")
    print("-" * 60)
    
    for persona, record, validation in results:
        print(f"{persona:<15} {record.turns:<6} {record.get_duration():<10.1f} {record.success:<8} {validation['score']:<8.2f}")

def demo_comprehensive_scenario_overview():
    """Show a comprehensive overview of all scenarios"""
    print("📊 === COMPREHENSIVE SCENARIO OVERVIEW ===\n")
    
    scenarios = ConversationScenarios.get_all_scenarios()
    personas = get_available_personas()
    
    print(f"Total Scenarios: {len(scenarios)}")
    print(f"Total Personas: {len(personas)}")
    
    # Scenario statistics
    scenario_stats = {}
    for scenario_id, scenario in scenarios.items():
        tags = scenario.tags
        for tag in tags:
            scenario_stats[tag] = scenario_stats.get(tag, 0) + 1
    
    print(f"\n🏷️ Scenario Categories:")
    for tag, count in sorted(scenario_stats.items()):
        print(f"   {tag}: {count} scenarios")
    
    # Persona coverage
    print(f"\n🎭 Persona Coverage:")
    for persona in personas:
        compatible_scenarios = ScenarioMatcher.get_scenarios_for_persona(persona)
        primary_scenarios = [s for s in compatible_scenarios if s.primary_persona == persona]
        print(f"   {persona}: {len(compatible_scenarios)} total, {len(primary_scenarios)} primary")
    
    # Expected conversation lengths
    print(f"\n⏱️ Expected Conversation Lengths:")
    length_categories = {
        "Quick (1-3 turns)": 0,
        "Medium (4-6 turns)": 0, 
        "Long (7+ turns)": 0
    }
    
    for scenario in scenarios.values():
        max_turns = scenario.estimated_turns[1]
        if max_turns <= 3:
            length_categories["Quick (1-3 turns)"] += 1
        elif max_turns <= 6:
            length_categories["Medium (4-6 turns)"] += 1
        else:
            length_categories["Long (7+ turns)"] += 1
    
    for category, count in length_categories.items():
        print(f"   {category}: {count} scenarios")

def interactive_scenario_explorer():
    """Interactive exploration of scenarios and personas"""
    print("🧭 === INTERACTIVE SCENARIO EXPLORER ===\n")
    
    while True:
        print("\nChoose an option:")
        print("1. View all scenarios")
        print("2. View persona-scenario mapping")
        print("3. Run single scenario demo")
        print("4. Compare personas in same scenario")
        print("5. Comprehensive overview")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                show_all_scenarios()
            elif choice == "2":
                show_persona_scenario_mapping()
            elif choice == "3":
                demo_single_scenario()
            elif choice == "4":
                demo_persona_scenario_combinations()
            elif choice == "5":
                demo_comprehensive_scenario_overview()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice! Please select 1-6.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 === CONVERSATION SCENARIO DEMO SYSTEM ===\n")
    
    print("This demo showcases the conversation scenario system that tests")
    print("all possible agent flows with different user personas.\n")
    
    try:
        interactive_scenario_explorer()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted! Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Make sure you have the environment properly configured.") 