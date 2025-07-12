"""
Demo script to showcase different persona behaviors in simulated conversations.
This demonstrates how each persona responds to typical supervisor messages.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.user_simulator.simulator import UserSimulator, get_available_personas
from tests.user_simulator.personas import PersonaDefinitions

def demo_persona_initial_responses():
    """Demonstrate how each persona responds to an initial greeting"""
    print("🎭 === PERSONA INITIAL RESPONSES DEMO ===\n")
    
    supervisor_greeting = """Hello! Welcome to our Python Developer position discussion. 
I'm here to help you learn about this exciting opportunity at our tech company. 
Are you interested in learning more about this role?"""
    
    personas = get_available_personas()
    
    for persona_name in personas:
        print(f"👤 **{persona_name.upper()} CANDIDATE**")
        print(f"   Temperature: {PersonaDefinitions.get_all_personas()[persona_name].temperature}")
        
        try:
            # Create simulator
            simulator = UserSimulator(persona_name)
            
            # Get initial response
            initial_response = simulator.start_conversation()
            print(f"   Initial: \"{initial_response}\"")
            
            # Respond to supervisor greeting
            response = simulator.respond_to_supervisor(supervisor_greeting)
            print(f"   Response: \"{response}\"")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print()

def demo_persona_to_job_question():
    """Demonstrate how each persona responds to job information"""
    print("🎭 === PERSONA JOB QUESTION RESPONSES ===\n")
    
    job_info = """This Python Developer role involves developing and maintaining high-quality software, 
working with data pipelines, and collaborating with cross-functional teams. 
You'll be using frameworks like Django, NumPy, Pandas, and working with cloud platforms like AWS. 
The position requires 3+ years of Python experience and offers great growth opportunities."""
    
    personas = get_available_personas()
    
    for persona_name in personas:
        print(f"👤 **{persona_name.upper()} CANDIDATE**")
        
        try:
            # Create simulator and start conversation
            simulator = UserSimulator(persona_name)
            simulator.start_conversation()
            
            # Respond to job information
            response = simulator.respond_to_supervisor(job_info)
            print(f"   Response: \"{response}\"")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print()

def demo_persona_scheduling_responses():
    """Demonstrate how each persona responds to scheduling requests"""
    print("🎭 === PERSONA SCHEDULING RESPONSES ===\n")
    
    scheduling_message = """Based on our conversation, I think you'd be a great fit for this role! 
Would you like to schedule an interview? I have availability next week in the mornings and afternoons. 
What works best for your schedule?"""
    
    personas = get_available_personas()
    
    for persona_name in personas:
        print(f"👤 **{persona_name.upper()} CANDIDATE**")
        
        try:
            # Create simulator and start conversation
            simulator = UserSimulator(persona_name)
            simulator.start_conversation()
            
            # First, give them some job info to build context
            simulator.respond_to_supervisor("This is a Python developer role with great opportunities.")
            
            # Then ask about scheduling
            response = simulator.respond_to_supervisor(scheduling_message)
            print(f"   Response: \"{response}\"")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print()

def demo_persona_characteristics():
    """Show the key characteristics of each persona"""
    print("🎭 === PERSONA CHARACTERISTICS OVERVIEW ===\n")
    
    personas = PersonaDefinitions.get_all_personas()
    
    for persona_name, config in personas.items():
        print(f"👤 **{config.name.upper()}**")
        print(f"   Temperature: {config.temperature}")
        print(f"   Conversation Length: {config.conversation_length}")
        print(f"   Scheduling Behavior: {config.scheduling_behavior}")
        print(f"   Exit Likelihood: {config.exit_likelihood}")
        print(f"   Sample Questions: {config.question_preferences[:2]}")
        print(f"   Typical Responses: {config.typical_responses[:2]}")
        print()

def interactive_persona_demo():
    """Interactive demo where user can test specific persona responses"""
    print("🎭 === INTERACTIVE PERSONA DEMO ===\n")
    
    available_personas = get_available_personas()
    print("Available personas:")
    for i, persona in enumerate(available_personas, 1):
        print(f"{i}. {persona}")
    
    try:
        choice = int(input("\nSelect persona number: ")) - 1
        if choice < 0 or choice >= len(available_personas):
            print("Invalid choice!")
            return
        
        persona_name = available_personas[choice]
        print(f"\n🎭 Testing {persona_name} persona...")
        
        # Create simulator
        simulator = UserSimulator(persona_name)
        initial_response = simulator.start_conversation()
        print(f"Candidate: {initial_response}")
        
        while True:
            supervisor_message = input("\nSupervisor message (or 'quit' to exit): ")
            if supervisor_message.lower() == 'quit':
                break
            
            user_response = simulator.respond_to_supervisor(supervisor_message)
            if user_response is None:
                print("Candidate: [Conversation ended]")
                break
            else:
                print(f"Candidate: {user_response}")
        
        # Show conversation summary
        summary = simulator.get_conversation_summary()
        print(f"\n📊 Conversation Summary:")
        print(f"   Turns: {summary.get('turn_count', 0)}")
        print(f"   Duration: {summary.get('duration_seconds', 0):.1f} seconds")
        print(f"   Expressed Interest: {summary.get('expressed_interest', False)}")
        print(f"   Asked Questions: {summary.get('asked_questions', False)}")
        print(f"   Discussed Scheduling: {summary.get('scheduling_discussed', False)}")
        
    except ValueError:
        print("Please enter a valid number!")
    except KeyboardInterrupt:
        print("\nDemo interrupted!")

if __name__ == "__main__":
    print("🚀 === USER SIMULATOR PERSONA DEMO ===\n")
    
    print("Choose demo mode:")
    print("1. Initial responses comparison")
    print("2. Job information responses")  
    print("3. Scheduling responses")
    print("4. Persona characteristics overview")
    print("5. Interactive persona testing")
    print("6. Run all demos")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            demo_persona_initial_responses()
        elif choice == "2":
            demo_persona_to_job_question()
        elif choice == "3":
            demo_persona_scheduling_responses()
        elif choice == "4":
            demo_persona_characteristics()
        elif choice == "5":
            interactive_persona_demo()
        elif choice == "6":
            demo_persona_characteristics()
            demo_persona_initial_responses()
            demo_persona_to_job_question()
            demo_persona_scheduling_responses()
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted! Goodbye! 👋")
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Make sure you have the required dependencies installed and environment variables set.") 