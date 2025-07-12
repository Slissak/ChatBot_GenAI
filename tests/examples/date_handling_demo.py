"""
Date Handling Demo - Demonstrates the comprehensive date parsing and generation
system for scheduling scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import date, timedelta
from tests.user_simulator.date_handler import (
    DateParser, DateGenerator, ConversationDateContext,
    parse_date_from_text, generate_test_dates, format_date_for_scheduling,
    DateType, TimeFrame
)

def demo_date_parsing():
    """Demonstrate date parsing capabilities"""
    print("🗓️ === DATE PARSING DEMO ===\n")
    
    # Set a fixed base date for consistent testing
    base_date = date(2025, 1, 9)  # Thursday
    parser = DateParser(base_date)
    
    print(f"Base date for testing: {base_date.strftime('%A, %B %d, %Y')}\n")
    
    # Test various date expressions
    test_expressions = [
        # Relative dates
        "tomorrow",
        "next week", 
        "next Monday",
        "next Friday",
        "in 3 days",
        "in 2 weeks",
        "tomorrow morning",
        "next Tuesday afternoon",
        
        # Specific dates
        "2025-01-15",
        "01/20/2025",
        "January 25, 2025",
        "Jan 30 2025",
        "15th January 2025",
        "February 5th in the morning",
        
        # Edge cases
        "today",
        "this Friday",
        "next week anytime"
    ]
    
    print("📅 Testing Date Expressions:")
    print("-" * 60)
    
    for expression in test_expressions:
        parsed = parser.parse_date(expression)
        if parsed:
            print(f"'{expression:20}' → {parsed.parsed_date} ({parsed.date_type.value})")
            if parsed.time_frame:
                print(f"{' '*23}   Time: {parsed.time_frame.value}")
            if not parsed.is_business_day:
                print(f"{' '*23}   ⚠️ Weekend day")
        else:
            print(f"'{expression:20}' → ❌ Could not parse")
        print()

def demo_date_generation():
    """Demonstrate date generation for scenarios"""
    print("\n🎯 === DATE GENERATION DEMO ===\n")
    
    base_date = date(2025, 1, 9)
    generator = DateGenerator(base_date)
    
    print(f"Generating dates from base: {base_date}\n")
    
    # Generate business dates
    print("📈 Future Business Dates:")
    business_dates = generator.generate_future_business_dates(5)
    for i, bdate in enumerate(business_dates, 1):
        print(f"  {i}. {bdate} ({bdate.strftime('%A')})")
    
    # Generate relative expressions
    print("\n🔄 Relative Date Expressions:")
    relative_expressions = generator.generate_relative_date_expressions()
    for expr in relative_expressions[:6]:  # Show first 6
        print(f"  • {expr}")
    
    # Generate specific expressions
    print("\n📋 Specific Date Expressions:")
    specific_expressions = generator.generate_specific_date_expressions(3)
    for expr in specific_expressions[:8]:  # Show first 8
        print(f"  • {expr}")

def demo_scenario_date_generation():
    """Demonstrate scenario-specific date generation"""
    print("\n🎬 === SCENARIO DATE GENERATION ===\n")
    
    base_date = date(2025, 1, 9)
    generator = DateGenerator(base_date)
    
    scenarios = [
        "immediate_scheduling",
        "flexible_scheduling", 
        "specific_date_request",
        "default"
    ]
    
    for scenario in scenarios:
        print(f"📋 Scenario: {scenario}")
        dates = generator.generate_scenario_dates(scenario)
        
        print(f"  Preferred: {dates['preferred_date']}")
        print(f"  Time Frame: {dates['time_frame']}")
        
        alternatives = dates['alternatives']
        if isinstance(alternatives, list) and alternatives:
            if isinstance(alternatives[0], date):
                alt_strs = [alt.strftime('%Y-%m-%d') for alt in alternatives[:3]]
            else:
                alt_strs = alternatives[:3]
            print(f"  Alternatives: {', '.join(map(str, alt_strs))}")
        
        print()

def demo_conversation_context():
    """Demonstrate conversation date context management"""
    print("\n💬 === CONVERSATION CONTEXT DEMO ===\n")
    
    base_date = date(2025, 1, 9)
    context = ConversationDateContext(base_date)
    
    # Simulate multiple conversations with different date requests
    conversations = [
        ("conv-001", "tomorrow afternoon"),
        ("conv-002", "next Monday"),
        ("conv-003", "January 20, 2025"),
        ("conv-004", "in 5 days"),
        ("conv-005", "next week anytime")
    ]
    
    print("🗂️ Setting conversation dates:")
    for conv_id, date_text in conversations:
        parsed = context.set_conversation_date(conv_id, date_text)
        if parsed:
            agent_format = context.format_for_agent(parsed)
            user_format = context.format_for_user(parsed)
            
            print(f"  {conv_id}: '{date_text}'")
            print(f"    Agent format: {agent_format}")
            print(f"    User format: {user_format}")
            
            # Show alternatives
            alternatives = context.suggest_alternatives(parsed, 2)
            alt_strs = [alt.strftime('%m/%d') for alt in alternatives]
            print(f"    Alternatives: {', '.join(alt_strs)}")
            print()

def demo_integration_with_scheduling():
    """Demonstrate integration with scheduling agent"""
    print("\n🤝 === SCHEDULING INTEGRATION DEMO ===\n")
    
    # Simulate user inputs that need to be processed for scheduling
    user_inputs = [
        "Can we schedule for tomorrow morning?",
        "I'm available next Tuesday afternoon",
        "How about January 15th?",
        "Next week would work for me",
        "I prefer Friday anytime"
    ]
    
    print("🔄 Processing user inputs for scheduling:")
    print("-" * 50)
    
    for user_input in user_inputs:
        print(f"User: '{user_input}'")
        
        # Extract and format date for scheduling agent
        formatted_date = format_date_for_scheduling(user_input)
        
        if formatted_date:
            print(f"  → Extracted date: {formatted_date}")
            
            # Parse to get additional context
            parsed = parse_date_from_text(user_input)
            if parsed and parsed.time_frame:
                print(f"  → Time preference: {parsed.time_frame.value}")
            
            print(f"  → Ready for sched_agent: date='{formatted_date}'")
            if parsed and parsed.time_frame and parsed.time_frame != TimeFrame.ANYTIME:
                print(f"      time_frame='{parsed.time_frame.value}'")
        else:
            print("  → ❌ No date found - ask for clarification")
        
        print()

def demo_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n⚠️ === EDGE CASES DEMO ===\n")
    
    base_date = date(2025, 1, 9)  # Thursday
    parser = DateParser(base_date)
    
    edge_cases = [
        "next Sunday",  # Weekend
        "yesterday",    # Past date
        "this Thursday", # Same day as base
        "13/25/2025",   # Invalid date
        "February 30th", # Invalid date
        "next Moonday",  # Typo
        "sometime next week", # Vague
        "as soon as possible", # Vague
        ""              # Empty
    ]
    
    print("🧪 Testing edge cases:")
    print("-" * 40)
    
    for case in edge_cases:
        if not case:
            print("Empty string:")
        else:
            print(f"'{case}':")
        
        try:
            parsed = parser.parse_date(case)
            if parsed:
                print(f"  ✅ Parsed: {parsed.parsed_date}")
                if not parsed.is_business_day:
                    print(f"  ⚠️ Warning: Weekend day")
            else:
                print(f"  ❌ Could not parse")
        except Exception as e:
            print(f"  💥 Error: {e}")
        
        print()

def demo_business_logic():
    """Demonstrate business logic and recommendations"""
    print("\n💼 === BUSINESS LOGIC DEMO ===\n")
    
    base_date = date(2025, 1, 9)  # Thursday
    context = ConversationDateContext(base_date)
    
    # Test business scenarios
    scenarios = [
        ("User requests weekend", "next Sunday"),
        ("User requests past date", "yesterday"),
        ("User requests today", "today"),
        ("User requests far future", "in 3 months"),
        ("User requests holiday", "December 25, 2025")
    ]
    
    print("🔍 Business logic analysis:")
    print("-" * 45)
    
    for scenario_name, date_text in scenarios:
        print(f"{scenario_name}:")
        print(f"  Input: '{date_text}'")
        
        parsed = context.parser.parse_date(date_text)
        if parsed:
            print(f"  Parsed: {parsed.parsed_date} ({parsed.parsed_date.strftime('%A')})")
            
            # Business logic checks
            if not parsed.is_business_day:
                print("  🚨 Issue: Weekend day - suggest alternative business day")
                alternatives = context.suggest_alternatives(parsed, 1)
                if alternatives:
                    print(f"  💡 Suggestion: {alternatives[0]} ({alternatives[0].strftime('%A')})")
            
            if parsed.parsed_date < base_date:
                print("  🚨 Issue: Past date - cannot schedule")
                print("  💡 Suggestion: Ask for future date")
            
            days_ahead = (parsed.parsed_date - base_date).days
            if days_ahead > 90:
                print("  ⚠️ Warning: Far future date - confirm availability")
            elif days_ahead == 0:
                print("  ⚠️ Warning: Same day - check immediate availability")
            
        else:
            print("  ❌ Could not parse - ask for clarification")
        
        print()

def comprehensive_demo():
    """Run the complete date handling demonstration"""
    print("🚀 === COMPREHENSIVE DATE HANDLING DEMO ===\n")
    
    print("This demonstration showcases:")
    print("• Natural language date parsing")
    print("• Relative and specific date handling") 
    print("• Date generation for testing scenarios")
    print("• Conversation context management")
    print("• Integration with scheduling systems")
    print("• Business logic and edge case handling\n")
    
    try:
        demo_date_parsing()
        demo_date_generation()
        demo_scenario_date_generation()
        demo_conversation_context()
        demo_integration_with_scheduling()
        demo_edge_cases()
        demo_business_logic()
        
        print("\n🎉 === DEMO COMPLETE ===")
        print("\nThe date handling system provides:")
        print("✅ Comprehensive natural language date parsing")
        print("✅ Scenario-appropriate date generation")
        print("✅ Conversation context management")
        print("✅ Scheduling agent integration")
        print("✅ Business logic validation")
        print("✅ Edge case handling")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_demo() 