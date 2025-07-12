"""
UserSimulator class - the main interface for simulating user conversations
with the supervisor agent using different personas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Optional, List, Dict, Tuple
import uuid
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .personas import PersonaDefinitions, PersonaConfig, PersonaResponseGenerator
from .date_utils import DateGenerator

class ConversationContext:
    """Tracks the state and context of an ongoing conversation"""
    
    def __init__(self, conversation_id: str, persona_name: str):
        self.conversation_id = conversation_id
        self.persona_name = persona_name
        self.messages: List[Dict[str, str]] = []
        self.turn_count = 0
        self.has_scheduled = False
        self.has_asked_questions = False
        self.expressed_interest = False
        self.scheduling_discussed = False
        self.conversation_ended = False
        self.start_time = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": self.turn_count
        })
        
        if role == "user":
            self.turn_count += 1
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history for context"""
        history = []
        for msg in self.messages:
            role = "Candidate" if msg["role"] == "user" else "Assistant"
            history.append(f"{role}: {msg['content']}")
        return "\n".join(history)

class UserSimulator:
    """
    Main class for simulating user conversations with different personas.
    Uses LLM to generate realistic responses based on persona characteristics.
    """
    
    def __init__(self, persona_name: str = "eager", model_name: str = "gpt-4o"):
        """
        Initialize the user simulator with a specific persona
        
        Args:
            persona_name: Name of the persona to use ('eager', 'skeptical', etc.)
            model_name: LLM model to use for response generation
        """
        self.logger = logging.getLogger(f'user_simulator_{persona_name}')
        
        # Load persona configuration
        personas = PersonaDefinitions.get_all_personas()
        if persona_name not in personas:
            raise ValueError(f"Unknown persona: {persona_name}. Available: {list(personas.keys())}")
        
        self.persona_config = personas[persona_name]
        self.persona_name = persona_name
        
        # Initialize LLM with persona-specific temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=self.persona_config.temperature,
            streaming=False
        )
        
        # Initialize persona response generator
        self.response_generator = PersonaResponseGenerator(self.persona_config)
        
        # Initialize date generator for scheduling
        self.date_generator = DateGenerator()
        
        # Current conversation context
        self.current_conversation: Optional[ConversationContext] = None
        
        self.logger.info(f"🎭 UserSimulator initialized with persona: {self.persona_config.name}")
        self.logger.info(f"🌡️ Temperature: {self.persona_config.temperature}")
    
    def start_conversation(self, initial_message: str = "Hello") -> str:
        """
        Start a new conversation with the supervisor
        
        Args:
            initial_message: First message to send (default: "Hello")
            
        Returns:
            The user's initial response to the supervisor
        """
        conversation_id = str(uuid.uuid4())
        self.current_conversation = ConversationContext(conversation_id, self.persona_name)
        
        self.logger.info(f"🆕 Starting new conversation: {conversation_id}")
        self.logger.info(f"🎭 Using persona: {self.persona_config.name}")
        
        # Generate initial user message
        user_response = self._generate_initial_response()
        
        # Add to conversation history
        self.current_conversation.add_message("user", user_response)
        
        return user_response
    
    def respond_to_supervisor(self, supervisor_message: str) -> Optional[str]:
        """
        Generate user response to supervisor message based on persona
        
        Args:
            supervisor_message: Message from the supervisor agent
            
        Returns:
            User response or None if conversation should end
        """
        if not self.current_conversation:
            raise ValueError("No active conversation. Call start_conversation() first.")
        
        if self.current_conversation.conversation_ended:
            self.logger.info("💬 Conversation already ended, no response generated")
            return None
        
        # Add supervisor message to conversation
        self.current_conversation.add_message("supervisor", supervisor_message)
        
        # Check if conversation should end
        if "[END]" in supervisor_message:
            self.current_conversation.conversation_ended = True
            self.logger.info("🔚 Conversation ended by supervisor")
            return None
        
        # Update response generator state
        self.response_generator.conversation_turn = self.current_conversation.turn_count
        
        # Check if persona wants to exit early
        if self.response_generator.should_exit():
            self.current_conversation.conversation_ended = True
            self.logger.info(f"🚪 {self.persona_config.name} decided to exit conversation")
            return self._generate_exit_response()
        
        # Generate response based on supervisor message and persona
        user_response = self._generate_response_to_supervisor(supervisor_message)
        
        # Add user response to conversation
        self.current_conversation.add_message("user", user_response)
        
        return user_response
    
    def _generate_initial_response(self) -> str:
        """Generate the initial user message based on persona"""
        # Use typical responses for initial greeting
        if self.persona_config.typical_responses:
            import random
            return random.choice(self.persona_config.typical_responses)
        else:
            return "Hello, I'm interested in the Python developer position."
    
    def _generate_response_to_supervisor(self, supervisor_message: str) -> str:
        """Generate contextual response based on supervisor message and persona"""
        
        # Analyze supervisor message content
        message_analysis = self._analyze_supervisor_message(supervisor_message)
        
        # Build prompt for LLM based on persona and context
        prompt_context = self._build_response_prompt(supervisor_message, message_analysis)
        
        # Generate response using LLM
        try:
            messages = [
                SystemMessage(content=self.persona_config.system_prompt),
                HumanMessage(content=prompt_context)
            ]
            
            response = self.llm.invoke(messages)
            user_response = response.content.strip()
            
            # Post-process response to ensure it matches persona
            user_response = self._post_process_response(user_response, message_analysis)
            
            self.logger.debug(f"💬 Generated response: {user_response[:100]}...")
            return user_response
            
        except Exception as e:
            self.logger.error(f"❌ Error generating response: {e}")
            # Fallback to typical response
            if self.persona_config.typical_responses:
                import random
                return random.choice(self.persona_config.typical_responses)
            return "I see. Can you tell me more?"
    
    def _analyze_supervisor_message(self, message: str) -> Dict[str, bool]:
        """Analyze the supervisor message to understand its content"""
        message_lower = message.lower()
        
        return {
            "contains_greeting": any(word in message_lower for word in ["hello", "hi", "welcome", "greet"]),
            "contains_job_info": any(word in message_lower for word in ["position", "role", "job", "develop", "python"]),
            "contains_question_answer": any(word in message_lower for word in ["experience", "skills", "technologies", "responsibilities"]),
            "contains_scheduling": any(word in message_lower for word in ["schedule", "interview", "appointment", "meet", "available"]),
            "contains_slots": any(word in message_lower for word in ["slots", "time", "date", "morning", "afternoon"]),
            "asks_for_interest": any(phrase in message_lower for phrase in ["interested", "would you like", "are you"]),
            "asks_for_questions": any(phrase in message_lower for phrase in ["questions", "anything else", "more information"]),
        }
    
    def _build_response_prompt(self, supervisor_message: str, analysis: Dict[str, bool]) -> str:
        """Build the prompt for LLM response generation"""
        
        conversation_history = self.current_conversation.get_conversation_history()
        turn_number = self.current_conversation.turn_count
        
        prompt = f"""
CONVERSATION CONTEXT:
Turn Number: {turn_number}
Conversation History:
{conversation_history}

CURRENT SUPERVISOR MESSAGE:
{supervisor_message}

MESSAGE ANALYSIS:
- Contains greeting: {analysis['contains_greeting']}
- Contains job information: {analysis['contains_job_info']}
- Contains scheduling discussion: {analysis['contains_scheduling']}
- Asks for your interest: {analysis['asks_for_interest']}
- Asks if you have questions: {analysis['asks_for_questions']}

RESPONSE INSTRUCTIONS:
Based on your persona characteristics, generate a natural response to the supervisor's message.

SPECIFIC GUIDANCE:
"""
        
        # Add persona-specific guidance based on message analysis
        if analysis['asks_for_interest'] and not self.current_conversation.expressed_interest:
            if self.persona_config.scheduling_behavior == "not_interested":
                prompt += "- Express polite disinterest and decline the opportunity\n"
            elif self.persona_config.scheduling_behavior == "immediately_interested":
                prompt += "- Express high enthusiasm and immediate interest\n"
            else:
                prompt += "- Express appropriate level of interest based on your persona\n"
        
        if analysis['contains_job_info'] and self.response_generator.should_ask_question():
            prompt += f"- Ask one relevant question from your preferences: {self.persona_config.question_preferences[:3]}\n"
        
        if analysis['contains_scheduling']:
            if self.response_generator.should_schedule(supervisor_message):
                # Include date preference
                date_preference = self.date_generator.get_random_date_request()
                prompt += f"- Express interest in scheduling and suggest: {date_preference}\n"
            elif self.persona_config.scheduling_behavior == "hesitant":
                prompt += "- Express uncertainty about scheduling, ask for time to think\n"
            elif self.persona_config.scheduling_behavior == "not_interested":
                prompt += "- Politely decline scheduling\n"
        
        prompt += "\nGenerate a response that is natural, conversational, and matches your persona exactly. Keep it concise and realistic."
        
        return prompt
    
    def _post_process_response(self, response: str, analysis: Dict[str, bool]) -> str:
        """Post-process the generated response to ensure consistency"""
        
        # Update conversation state based on response content
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["interested", "excited", "love to"]):
            self.current_conversation.expressed_interest = True
        
        if any(word in response_lower for word in ["schedule", "interview", "meet"]):
            self.current_conversation.scheduling_discussed = True
        
        if "?" in response:
            self.current_conversation.has_asked_questions = True
        
        # Ensure response length matches persona
        if self.persona_config.conversation_length == "very_short" and len(response) > 100:
            # Truncate for very short personas
            sentences = response.split('. ')
            response = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
        
        return response
    
    def _generate_exit_response(self) -> str:
        """Generate appropriate exit response for persona"""
        exit_responses = {
            "disinterested": "Thank you for your time, but I'm not interested in this opportunity.",
            "direct": "I don't think this is the right fit for me. Thanks anyway.",
            "skeptical": "I need to think about this more. I'll be in touch if I'm interested.",
            "indecisive": "I'm not sure this is the right time for me to make a career change.",
            "eager": "Thank you so much! I look forward to hearing from you soon.",
            "detail_oriented": "Thank you for all the information. I'll review everything and get back to you."
        }
        
        return exit_responses.get(self.persona_name, "Thank you for your time.")
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of the current conversation"""
        if not self.current_conversation:
            return {}
        
        return {
            "conversation_id": self.current_conversation.conversation_id,
            "persona": self.persona_name,
            "turn_count": self.current_conversation.turn_count,
            "message_count": len(self.current_conversation.messages),
            "duration_seconds": (datetime.now() - self.current_conversation.start_time).total_seconds(),
            "expressed_interest": self.current_conversation.expressed_interest,
            "has_scheduled": self.current_conversation.has_scheduled,
            "asked_questions": self.current_conversation.has_asked_questions,
            "scheduling_discussed": self.current_conversation.scheduling_discussed,
            "conversation_ended": self.current_conversation.conversation_ended,
            "messages": self.current_conversation.messages
        }
    
    def reset(self):
        """Reset the simulator for a new conversation"""
        self.current_conversation = None
        self.response_generator = PersonaResponseGenerator(self.persona_config)
        self.logger.info(f"🔄 UserSimulator reset for {self.persona_config.name}")

# Convenience functions for easy usage
def create_simulator(persona_name: str) -> UserSimulator:
    """Create a UserSimulator with the specified persona"""
    return UserSimulator(persona_name)

def get_available_personas() -> List[str]:
    """Get list of available persona names"""
    return list(PersonaDefinitions.get_all_personas().keys())

def simulate_conversation_turn(simulator: UserSimulator, supervisor_message: str) -> Tuple[Optional[str], bool]:
    """
    Simulate a single conversation turn
    
    Returns:
        (user_response, conversation_continues)
    """
    user_response = simulator.respond_to_supervisor(supervisor_message)
    conversation_continues = user_response is not None and not simulator.current_conversation.conversation_ended
    
    return user_response, conversation_continues 