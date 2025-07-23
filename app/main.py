import sys
import os
import logging
import uuid
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import our modules using relative imports
from app.logging_config import setup_logging, log_conversation_event, log_system_health
from app.agents.agents import SupervisorAgent

def main():
    # Setup main logger
    log_dir = os.path.join(project_root, 'logs')
    setup_logging(log_level=logging.DEBUG, log_dir=log_dir)
    logger = logging.getLogger('main_app') # Use a specific logger name
    logger.info("🚀 === Python Developer Job Assistant Starting ===")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize supervisor with initial conversation ID
    initial_conversation_id = str(uuid.uuid4())
    logger.info(f"📋 Initial Session ID: {initial_conversation_id}")
    
    try:
        # Initialize the supervisor agent
        supervisor = SupervisorAgent(conversation_id=initial_conversation_id)
        logger.info("✅ System initialization completed successfully")
        
        # Start initial conversation
        initial_response = supervisor.process_message("Hello")
        print(f"Assistant: {initial_response}\n")
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                user_message = input("Enter your message: ")
                logger.debug(f"📝 User input received: {user_message[:50]}...")
                
                # Process the message
                response = supervisor.process_message(user_message)
                
                # Check if this is an end message

                if '[END]' in response:
                    # Remove the [END] marker for display
                    display_response = response.replace('[END]', '').strip()
                    print(f"Assistant: {display_response}\n")
                    
                    # Visual separator with emojis
                    separator = "="*50
                    print(f"\n{separator}")
                    print("🔚 Conversation session ended")
                    print("🔄 Starting new session...")
                    print(f"{separator}\n")
                    
                    # Start new conversation with fresh UUID
                    new_id = supervisor.start_new_conversation()
                    logger.info(f"🆕 New conversation session started: {new_id}")
                    
                    # Reset exit decision
                    supervisor.last_exit_decision = None
                    
                    # Trigger natural greeting for new conversation
                    new_response = supervisor.process_message("Hello")
                    print(f"Assistant: {new_response}\n")
                else:
                    print(f"Assistant: {response}\n")
                    
            except KeyboardInterrupt:
                logger.info("👋 User initiated shutdown")
                log_conversation_event(supervisor.conversation_id, "ended", {"reason": "user_exit"})
                print("\n\n👋 Goodbye! Exiting the conversation system.")
                break
            except Exception as e:
                logger.error(f"❌ Error in conversation loop: {str(e)}")
                log_system_health("Main Loop", "ERROR", {"error": str(e)})
                print(f"\n❌ Error occurred: {str(e)}")
                print("🔄 Continuing with current session...")
                
    except Exception as e:
        logger.critical(f"💥 Critical error during system initialization: {str(e)}")
        log_system_health("System Initialization", "ERROR", {"error": str(e)})
        print(f"💥 Failed to initialize system: {str(e)}")
        return
    
    logger.info("🏁 Application shutdown complete")

if __name__ == "__main__":
    main()

