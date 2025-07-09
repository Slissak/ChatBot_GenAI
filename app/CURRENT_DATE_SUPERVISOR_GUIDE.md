# Adding Current Date to LangGraph Supervisor Agent - Comprehensive Guide

## Overview

For scheduling functionality, the supervisor agent needs access to the current date to make informed decisions about interview scheduling. This guide shows multiple proven approaches based on official LangGraph documentation and best practices.

## Approach 1: Dynamic System Prompt with `state_modifier` ⭐ **RECOMMENDED**

Based on official LangGraph documentation, the best approach is using a `state_modifier` function that dynamically injects current date information.

### Implementation for `create_react_agent`:

```python
from datetime import datetime

def date_aware_state_modifier(state):
    """Dynamically inject current date into system prompt"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    system_prompt = f"""
You are a helpful hiring assistant for the Python developer position.

CURRENT DATE INFORMATION:
- Today's date: {current_date}
- Current date and time: {current_datetime}
- Current year: {datetime.now().year}

Use this date information when scheduling interviews or making date-related decisions.

CONVERSATION FLOW:
- Start conversations with a warm greeting, introduce yourself, 
- ask for their full name, and ask if they're interested in the Python developer position
- Afterwards, suggest to them to schedule an interview or ask them if they have any questions about the job
- You aim to set an interview with the candidate so direct them into scheduling an interview
- After scheduling an interview, summarize the details of the interview information and ask if they need anything else
- You should use the exit_agent if you already scheduled an interview or understand that the candidate is not interested in the job or already found a job
- Don't insert [END] to the conversation by yourself, Let the exit_agent decide.
- If the candidate asking questions about the job DONT ANSWER YOUSELF, use the info_agent to answer the question.

CRITICAL: When agents return to you with responses, you MUST include their full response in your answer to the user. 
Do not ignore agent responses - they contain the information the user requested.

You have 3 agents at your disposal: "info_agent", "sched_agent", and "exit_agent".

info_agent: whenever the candidate wants to inquire about the job you should use this agent. Extract the question from the candidate's
message and then use the info_agent to answer the question. 
IMPORTANT: When the info_agent provides an answer, you MUST relay that complete answer to the user, then ask if they have any other questions or want to schedule an interview.

sched_agent: whenever the candidate wants to schedule an interview you should use this agent.
ALWAYS provide the agent with at least the date and maybe the preferred time frame if given. 
The agent should return available slots and suggest other slots if no available slots are found.
IMPORTANT: When the sched_agent provides scheduling information, you MUST relay that complete information to the user.

exit_agent: this agent is used to determine if the conversation should end. 
Send the full conversation history to the agent and let it decide if the conversation should end.
The agent will return only "END" and "NOT END" as the result, 
if the result is "END" then you should end the conversation by only printing [END] to the user,
if you got "NOT END" then you should continue the conversation.
"""
    
    return [{"role": "system", "content": system_prompt}] + state["messages"]

# Apply to create_react_agent
supervisor_agent = create_react_agent(
    model=self.llm,
    tools=handoff_tools,
    state_modifier=date_aware_state_modifier  # 🎯 Key change here
)
```

### Implementation for `create_supervisor`:

```python
from datetime import datetime
from langgraph_supervisor import create_supervisor

def get_current_date_prompt():
    """Generate supervisor prompt with current date"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')
    
    return f"""
You are a helpful hiring assistant for the Python developer position.

CURRENT DATE INFORMATION:
- Today's date: {current_date}
- Current time: {current_time}
- Use this information when making scheduling decisions

CONVERSATION FLOW:
[rest of your prompt...]
"""

# Create supervisor with dynamic date-aware prompt
workflow = create_supervisor(
    agents=[self.info_agent.agent, self.exit_agent.agent, self.sched_agent.agent],
    model=self.llm,
    prompt=get_current_date_prompt(),  # 🎯 Dynamic prompt with current date
    add_handoff_back_messages=True,
    output_mode="full_history"
)
```

## Approach 2: Agent State with Date Information

Add current date to the agent state that can be accessed by all agents:

```python
from typing import Annotated
from datetime import datetime
from langgraph.graph.message import add_messages

class DateAwareAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_date: str
    current_datetime: str

def supervisor_node(state: DateAwareAgentState):
    # Ensure current date is always updated
    state["current_date"] = datetime.now().strftime('%Y-%m-%d')
    state["current_datetime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    messages = [
        {"role": "system", "content": f"""
        You are a supervisor managing agents.
        Current date: {state["current_date"]}
        Current datetime: {state["current_datetime"]}
        
        [rest of your supervisor prompt...]
        """},
    ] + state["messages"]
    
    # Continue with supervisor logic...
```

## Approach 3: Tool-Based Current Date Access

Create a tool that agents can use to get current date/time:

```python
from langchain.tools import tool
from datetime import datetime

@tool
def get_current_datetime() -> str:
    """Get the current date and time for scheduling purposes"""
    current_datetime = datetime.now()
    return f"""
    Current Date: {current_datetime.strftime('%Y-%m-%d')}
    Current Time: {current_datetime.strftime('%H:%M:%S')}
    Day of Week: {current_datetime.strftime('%A')}
    """

# Add to your agents
supervisor_agent = create_react_agent(
    model=self.llm,
    tools=[get_current_datetime] + other_tools,
    prompt="You can use get_current_datetime tool to check current date and time."
)
```

## Approach 4: Configuration-Based Date Injection

Use LangGraph's configuration system to inject date information:

```python
def create_date_aware_supervisor(conversation_id: str = None):
    def supervisor_with_date(state, config):
        # Get current date from config or generate it
        current_date = config.get("configurable", {}).get("current_date") or datetime.now().strftime('%Y-%m-%d')
        
        system_prompt = f"""
        You are a hiring supervisor.
        Today's date: {current_date}
        [rest of prompt...]
        """
        
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        # Continue with supervisor logic...
    
    workflow = StateGraph(State)
    workflow.add_node("supervisor", supervisor_with_date)
    # ... add other nodes
    
    return workflow.compile()

# Usage with date configuration
app = create_date_aware_supervisor()
result = app.invoke(
    {"messages": [HumanMessage(content="Schedule interview")]}, 
    config={"configurable": {"current_date": datetime.now().strftime('%Y-%m-%d')}}
)
```

## Implementation for Your Current System

Based on your current supervisor setup, here's the recommended implementation:

```python
class SupervisorAgent:
    def __init__(self, model_name: str = "gpt-4o", conversation_id: str = None):
        # ... existing initialization code ...
        
        # Create date-aware system prompt
        self.system_supervisor_prompt = self._create_date_aware_prompt()
        
        # ... rest of initialization ...
    
    def _create_date_aware_prompt(self) -> str:
        """Create supervisor prompt with current date information"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M')
        current_day = datetime.now().strftime('%A')
        
        return f"""
You are a helpful hiring assistant for the Python developer position.

CURRENT DATE INFORMATION:
- Today's date: {current_date}
- Current time: {current_time}
- Day of week: {current_day}
- Current year: {datetime.now().year}

Important: Use this date information when working with the sched_agent for interview scheduling.

CONVERSATION FLOW:
- Start conversations with a warm greeting, introduce yourself, 
- ask for their full name, and ask if they're interested in the Python developer position
- Afterwards, suggest to them to schedule an interview or ask them if they have any questions about the job
- You aim to set an interview with the candidate so direct them into scheduling an interview
- After scheduling an interview, summarize the details of the interview information and ask if they need anything else
- You should use the exit_agent if you already scheduled an interview or understand that the candidate is not interested in the job or already found a job
- Don't insert [END] to the conversation by yourself, Let the exit_agent decide.
- If the candidate asking questions about the job DONT ANSWER YOUSELF, use the info_agent to answer the question.

CRITICAL: When agents return to you with responses, you MUST include their full response in your answer to the user. 
Do not ignore agent responses - they contain the information the user requested.

You have 3 agents at your disposal: "info_agent", "sched_agent", and "exit_agent".

info_agent: whenever the candidate wants to inquire about the job you should use this agent. Extract the question from the candidate's
message and then use the info_agent to answer the question. 
IMPORTANT: When the info_agent provides an answer, you MUST relay that complete answer to the user, then ask if they have any other questions or want to schedule an interview.

sched_agent: whenever the candidate wants to schedule an interview you should use this agent.
ALWAYS provide the agent with at least the date and maybe the preferred time frame if given. 
The agent should return available slots and suggest other slots if no available slots are found.
IMPORTANT: When the sched_agent provides scheduling information, you MUST relay that complete information to the user.

exit_agent: this agent is used to determine if the conversation should end. 
Send the full conversation history to the agent and let it decide if the conversation should end.
The agent will return only "END" and "NOT END" as the result, 
if the result is "END" then you should end the conversation by only printing [END] to the user,
if you got "NOT END" then you should continue the conversation.
"""

    def refresh_date_context(self):
        """Refresh the supervisor prompt with current date - call this for new conversations"""
        self.system_supervisor_prompt = self._create_date_aware_prompt()
        
        # Recreate the workflow with updated prompt
        workflow = create_supervisor(
            agents=[self.info_agent.agent, self.exit_agent.agent, self.sched_agent.agent],
            model=self.llm,
            prompt=self.system_supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history"
        )
        
        self.workflow = workflow.compile(name="supervisor_workflow", checkpointer=self.memory)
        
    def start_new_conversation(self) -> str:
        """Start a new conversation session with a fresh UUID and current date"""
        old_id = self.conversation_id
        log_conversation_event(old_id, "ended", {"reason": "new_conversation_started"})
        
        self.conversation_id = str(uuid.uuid4())
        log_conversation_event(self.conversation_id, "started", {"previous_id": old_id})
        
        # Refresh date context for new conversation
        self.refresh_date_context()
        
        self.logger.info(f"🆕 Starting new conversation session: {self.conversation_id}")
        
        return self.conversation_id
```

## Best Practices

1. **Use Approach 1 (state_modifier)** for most use cases - it's the officially recommended method
2. **Refresh date context** for each new conversation to ensure accuracy
3. **Include timezone information** if your application serves multiple timezones
4. **Log date information** for debugging and audit purposes
5. **Test date-sensitive scenarios** to ensure proper behavior

## Example Usage in Scheduling Context

```python
# User: "I want to schedule an interview for tomorrow"
# Supervisor with current date knowledge can now:
# 1. Know what "tomorrow" means relative to today
# 2. Pass accurate date information to sched_agent
# 3. Make informed scheduling decisions
```

## Key Benefits

- ✅ **Accurate scheduling**: Supervisor knows current date for relative date calculations
- ✅ **Context awareness**: All agents have access to current date information  
- ✅ **Dynamic updates**: Date information refreshes for each new conversation
- ✅ **LangGraph compatible**: Uses officially supported patterns
- ✅ **Logging friendly**: Current date context included in logs

---

**Recommended Implementation**: Use the state_modifier approach (Approach 1) as it's the most robust and officially supported method for dynamic prompt injection in LangGraph. 