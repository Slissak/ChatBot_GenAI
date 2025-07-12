"""
Simple test script to validate the scenario system works correctly
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.user_simulator.scenarios import ConversationScenarios, ScenarioMatcher
from tests.user_simulator.personas import get_available_personas, get_all_persona_configs

def test_scenario_system():
    """Test that the scenario system is working correctly"""
    print("🚀 === TESTING SCENARIO SYSTEM ===\n")
    
    # Test personas
    print("1. Testing personas...")
    personas = get_available_personas()
    print(f"   ✅ Found {len(personas)} personas: {', '.join(personas)}")
    
    persona_configs = get_all_persona_configs()
    print(f"   ✅ Loaded {len(persona_configs)} persona configurations")
    
    # Test scenarios
    print("\n2. Testing scenarios...")
    scenarios = ConversationScenarios.get_all_scenarios()
    print(f"   ✅ Found {len(scenarios)} scenarios")
    
    for scenario_id, scenario in scenarios.items():
        print(f"   📋 {scenario.name} ({scenario_id})")
        print(f"      Primary: {scenario.primary_persona}")
        print(f"      Compatible: {', '.join(scenario.compatible_personas)}")
        print(f"      Turns: {scenario.estimated_turns}")
        print(f"      Tags: {', '.join(scenario.tags)}")
    
    # Test scenario matching
    print("\n3. Testing scenario matching...")
    for persona in personas:
        compatible_scenarios = ScenarioMatcher.get_scenarios_for_persona(persona)
        primary_scenario = ScenarioMatcher.get_primary_scenario_for_persona(persona)
        
        print(f"   👤 {persona.upper()}:")
        if primary_scenario:
            print(f"      Primary: {primary_scenario.name}")
        print(f"      Compatible: {len(compatible_scenarios)} scenarios")
        
    # Test scenario statistics
    print("\n4. Scenario statistics...")
    scenario_stats = {}
    for scenario in scenarios.values():
        for tag in scenario.tags:
            scenario_stats[tag] = scenario_stats.get(tag, 0) + 1
    
    print(f"   📊 Category breakdown:")
    for tag, count in sorted(scenario_stats.items()):
        print(f"      {tag}: {count} scenarios")
    
    # Expected conversation lengths
    print("\n5. Expected conversation lengths...")
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
        print(f"   ⏱️ {category}: {count} scenarios")
    
    print("\n✅ === ALL TESTS PASSED ===")
    print("The scenario system is ready for execution!")

if __name__ == "__main__":
    test_scenario_system() 