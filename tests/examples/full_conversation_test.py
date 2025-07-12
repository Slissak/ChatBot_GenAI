"""
Full conversation test - demonstrates complete integration between UserSimulator and SupervisorAgent.
This shows how personas interact with the actual multi-agent system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.main import SupervisorAgent
from tests.user_simulator.simulator import UserSimulator
import logging

# Set up logging to see the conversation flow
logging.basicConfig(level=logging.INFO)

def test_eager_candidate_full_conversation():
    """Test a complete conversation with an eager candidate"""
    print("🚀 === FULL CONVERSATION TEST: EAGER CANDIDATE ===\n")
    
    try:
        # Initialize supervisor and user simulator
        print("🔧 Initializing supervisor agent...")
        supervisor = SupervisorAgent()
        
        print("🎭 Initializing eager candidate simulator...")
        user_sim = UserSimulator("eager")
        
        # Start conversation
        print("\n💬 === CONVERSATION BEGINS ===")
        
        # User initiates
        user_message = user_sim.start_conversation()
        print(f"👤 Candidate: {user_message}")
        
        # Supervisor responds
        supervisor_response = supervisor.process_message(user_message)
        print(f"🤖 Supervisor: {supervisor_response}")
        
        # Continue conversation for a few turns
        for turn in range(4):  # Limit to prevent infinite loops
            if "[END]" in supervisor_response:
                print("\n🔚 Conversation ended by supervisor")
                break
            
            # User responds to supervisor
            user_message = user_sim.respond_to_supervisor(supervisor_response)
            if user_message is None:
                print("\n🔚 Conversation ended by user")
                break
            
            print(f"👤 Candidate: {user_message}")
            
            # Supervisor responds to user
            supervisor_response = supervisor.process_message(user_message)
            print(f"🤖 Supervisor: {supervisor_response}")
        
        # Show conversation summary
        summary = user_sim.get_conversation_summary()
        print("\n📊 === CONVERSATION SUMMARY ===")
        print(f"Persona: {summary['persona']}")
        print(f"Total turns: {summary['turn_count']}")
        print(f"Messages exchanged: {summary['message_count']}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Expressed interest: {summary['expressed_interest']}")
        print(f"Asked questions: {summary['asked_questions']}")
        print(f"Discussed scheduling: {summary['scheduling_discussed']}")
        print(f"Conversation ended: {summary['conversation_ended']}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def test_skeptical_candidate_conversation():
    """Test conversation with skeptical candidate - should ask many questions"""
    print("\n🚀 === FULL CONVERSATION TEST: SKEPTICAL CANDIDATE ===\n")
    
    try:
        # Initialize agents
        supervisor = SupervisorAgent()
        user_sim = UserSimulator("skeptical")
        
        print("💬 === CONVERSATION BEGINS ===")
        
        # Start conversation
        user_message = user_sim.start_conversation()
        print(f"👤 Candidate: {user_message}")
        
        supervisor_response = supervisor.process_message(user_message)
        print(f"🤖 Supervisor: {supervisor_response}")
        
        # Continue for more turns since skeptical candidate asks more questions
        for turn in range(6):
            if "[END]" in supervisor_response:
                print("\n🔚 Conversation ended by supervisor")
                break
            
            user_message = user_sim.respond_to_supervisor(supervisor_response)
            if user_message is None:
                print("\n🔚 Conversation ended by user")
                break
            
            print(f"👤 Candidate: {user_message}")
            
            supervisor_response = supervisor.process_message(user_message)
            print(f"🤖 Supervisor: {supervisor_response}")
        
        # Show summary
        summary = user_sim.get_conversation_summary()
        print("\n📊 === CONVERSATION SUMMARY ===")
        print(f"Persona: {summary['persona']} - Should ask many questions")
        print(f"Total turns: {summary['turn_count']}")
        print(f"Asked questions: {summary['asked_questions']} (Expected: True)")
        print(f"Expressed interest: {summary['expressed_interest']}")
        
        print("\n✅ Skeptical candidate test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")

def test_disinterested_candidate_conversation():
    """Test conversation with disinterested candidate - should end quickly"""
    print("\n🚀 === FULL CONVERSATION TEST: DISINTERESTED CANDIDATE ===\n")
    
    try:
        supervisor = SupervisorAgent()
        user_sim = UserSimulator("disinterested")
        
        print("💬 === CONVERSATION BEGINS ===")
        
        user_message = user_sim.start_conversation()
        print(f"👤 Candidate: {user_message}")
        
        supervisor_response = supervisor.process_message(user_message)
        print(f"🤖 Supervisor: {supervisor_response}")
        
        # Should end quickly
        for turn in range(3):
            if "[END]" in supervisor_response:
                print("\n🔚 Conversation ended by supervisor")
                break
            
            user_message = user_sim.respond_to_supervisor(supervisor_response)
            if user_message is None:
                print("\n🔚 Conversation ended by user")
                break
            
            print(f"👤 Candidate: {user_message}")
            
            supervisor_response = supervisor.process_message(user_message)
            print(f"🤖 Supervisor: {supervisor_response}")
        
        summary = user_sim.get_conversation_summary()
        print("\n📊 === CONVERSATION SUMMARY ===")
        print(f"Persona: {summary['persona']} - Should end quickly")
        print(f"Total turns: {summary['turn_count']} (Expected: Low)")
        print(f"Conversation ended: {summary['conversation_ended']} (Expected: True)")
        
        print("\n✅ Disinterested candidate test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")

def quick_persona_showcase():
    """Quick showcase of all personas responding to same supervisor message"""
    print("\n🎭 === QUICK PERSONA SHOWCASE ===\n")
    
    supervisor_message = "Hello! Are you interested in our Python developer position?"
    
    personas = ["eager", "skeptical", "direct", "detail_oriented", "indecisive", "disinterested"]
    
    for persona_name in personas:
        try:
            user_sim = UserSimulator(persona_name)
            user_sim.start_conversation()  # Initialize conversation
            
            response = user_sim.respond_to_supervisor(supervisor_message)
            print(f"👤 {persona_name.upper()}: {response}")
            
        except Exception as e:
            print(f"❌ {persona_name}: Error - {e}")
    
    print("\n✅ Showcase completed!")

if __name__ == "__main__":
    print("🧪 === PERSONA INTEGRATION TESTS ===\n")
    
    print("Choose test:")
    print("1. Eager candidate full conversation")
    print("2. Skeptical candidate conversation") 
    print("3. Disinterested candidate conversation")
    print("4. Quick persona showcase")
    print("5. Run all tests")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            test_eager_candidate_full_conversation()
        elif choice == "2":
            test_skeptical_candidate_conversation()
        elif choice == "3":
            test_disinterested_candidate_conversation()
        elif choice == "4":
            quick_persona_showcase()
        elif choice == "5":
            quick_persona_showcase()
            test_eager_candidate_full_conversation()
            test_skeptical_candidate_conversation()
            test_disinterested_candidate_conversation()
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted! 👋")
    except Exception as e:
        print(f"\nError running tests: {e}")
        print("Make sure you have the environment properly configured.") 