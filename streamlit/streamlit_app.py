import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, date
import os
import sys
import time
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add both project root and app directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app_dir = os.path.join(project_root, 'app')
sys.path.extend([project_root, app_dir])

# Now we can import from app
from app.main import SupervisorAgent
from app.logging_config import setup_logging

# Set up logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(project_root, 'logs')
setup_logging(log_level=logging.INFO, log_dir=log_dir)

# Set page config
st.set_page_config(
    page_title="Tech app chat bot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize all session state variables at the start
if "supervisor" not in st.session_state:
    st.session_state.supervisor = SupervisorAgent()
    st.session_state.conversation_id = st.session_state.supervisor.conversation_id

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_log" not in st.session_state:
    st.session_state.show_log = False

if "last_log_update" not in st.session_state:
    st.session_state.last_log_update = time.time()

# Main title
st.title("🚀 Welcome to Tech! \nReady for your next chapter?")
st.markdown("---")

# Sidebar - navigation and live log
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a section:", ["main", "About"])

# Live log viewer in sidebar with improved performance
show_log = st.sidebar.checkbox("Show Live Agent Log", value=st.session_state.show_log)
if show_log:
    log_placeholder = st.sidebar.empty()
    
    # Set log path in the project's log directory
    log_path = os.path.join(log_dir, 'agent_communications.log')
    
    # Only update log if 2 seconds have passed since last update
    current_time = time.time()
    if current_time - st.session_state.last_log_update >= 2:
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()[-50:]  # Get last 50 lines
                    log_content = "".join(lines)
                log_placeholder.markdown("#### Live Agent Log")
                log_placeholder.code(log_content, language="text")
            except Exception as e:
                st.sidebar.error(f"Error reading log file: {str(e)}")
        else:
            log_placeholder.info("Log file not found. Please check if logging is properly configured.")
            
        st.session_state.last_log_update = current_time

# Add footer to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("*Built with ❤️ using Streamlit | Powered by AI*")
st.sidebar.markdown("*Contributors: Sivan Lissak, Itay Gefner, Bonnie Erez*")

if page == "main":
    # st.header("🏠 Main Chat Area")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the Job or ask to schedule an interview..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from supervisor
        with st.chat_message("assistant"):
            response = st.session_state.supervisor.process_message(prompt)
            
            # Check if this is an end message
            if '[END]' in response:  # Changed from endswith to consistent check
                # Remove the [END] marker for display
                display_response = response.replace('[END]', '').strip()
                st.markdown(display_response)
                
                # Add visual separator with emojis for better UX
                separator = "\n\n" + "="*25 + "\n🔚 Conversation session ended\n🔄 Starting new session...\n" + "="*50 + "\n\n"
                
                # Start new conversation but keep history
                new_id = st.session_state.supervisor.start_new_conversation()
                st.session_state.conversation_id = new_id
                
                # Log the new conversation creation
                logging.getLogger('supervisor_agent').info(f"🆕 New conversation started in Streamlit with UUID: {new_id}")
                
                # Add messages to chat history with improved visual separation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": display_response
                })
                st.session_state.messages.append({
                    "role": "system",
                    "content": separator
                })
                
                # Reset the exit decision since we're starting fresh
                st.session_state.supervisor.last_exit_decision = None
                
                # Trigger a new greeting
                new_response = st.session_state.supervisor.process_message("Hello")
                
                # Add the new greeting to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": new_response
                })
                st.markdown(new_response)
            else:
                # Regular message display
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
elif page == "About":
    st.header("ℹ️ About Tech Company")
    st.markdown("""
# Job Interview Bot

## Contributors
- Sivan Lissak
- Itay Gefner
- Bonnie Erez

## Overview
This application is a multi-agent system designed to assist candidates during the job application process for a Python Developer position. The system provides two main features:
1. Information Agent: Answers questions about the Python Developer role, requirements, and company details using RAG (Retrieval-Augmented Generation).
2. Scheduling Agent: Helps candidates schedule interviews by querying available time slots from a database.

Additionally, the system includes a hidden Exit Agent that monitors the conversation to determine appropriate ending points, enhancing the natural flow of interactions without direct user awareness.

## Project Structure
Below is a detailed explanation of the key files and directories in the project:

```
.
├── app/                            # Main application directory
│   ├── __init__.py                # Python package initializer
│   ├── agents/                    # Agent implementations
│   │   ├── __init__.py           # Agents package initializer
│   │   └── agents.py             # Core agent implementations
│   ├── utils/                     # Utility functions and helpers
│   │   ├── __init__.py           # Utils package initializer
│   │   └── utils.py              # Utility functions
│   ├── db_Tech.sql               # SQL script for database setup
│   ├── logging_config.py         # Logging system configuration
│   ├── main.py                   # Main application entry point
│   ├── modules/                  # Additional modules
│   │   ├── Bot atchitecutre/     # Bot's structural design
│   │   │   ├── bot_architecture.py # Bot architecture implementation
│   │   │   └── icons/           # UI icons for agents
│   │   ├── conversation_ending_data_shuffled.jsonl  # Training dataset
│   │   ├── end_convr_train.py    # BERT Model training script
│   │   └── train_openai_via_api.py # OpenAI API training script
│   └── PythonDeveloperJobDescription.pdf  # Job description data
├── logs/                         # Application logs directory
├── streamlit/                    # Streamlit UI components
│   └── streamlit_app.py         # Main Streamlit application
├── tmp/                         # Contains the local database
├── pyproject.toml              # Project metadata and dependencies
├── README.md                   # Project documentation
└── requirements.txt            # Python package dependencies
```

## Bot Architecture
The bot's architecture is a multi-agent system where a supervisor agent coordinates between specialized agents to handle different aspects of the job interview process.
    """)

    # Display the architecture image
    image_path = os.path.join(project_root, "app", "modules", "Bot atchitecutre", "job_interview_bot_-_multi-agent_system.png")
    if os.path.exists(image_path):
        st.image(image_path, caption="Bot Architecture", use_container_width=True)
    else:
        st.error(f"Image not found at: {image_path}")

    st.markdown("""
## Detailed Agent Explanations

### Supervisor Agent
The Supervisor Agent is implemented using LangChain. It acts as the central coordinator, initializing and managing the other agents. It routes user queries to the appropriate agent based on the context and ensures smooth flow of the conversation.

### Info Agent
The Info Agent is a ReAct (Reasoning and Acting) agent that utilizes a vector database combined with Retrieval-Augmented Generation (RAG). It handles queries related to general information about the job, company, or interview process by retrieving relevant data from the vector store and generating informed responses.

### Scheduling Agent
The Scheduling Agent is a ReAct agent that connects to an existing SQL server. It queries the database for available interview slots and assists in booking or rescheduling interviews based on user availability and preferences.

### Exit Agent
The Exit Agent is a fine-tuned OpenAI model trained on a custom dataset created specifically for this purpose. It predicts when the conversation should end, such as when all necessary information has been exchanged or when the user indicates they are done.

## Application Features

### Logging System
The system provides extensive logging capabilities. Logs are stored in the `app/logs/` directory, allowing you to track:
- Chat turns and conversation flow
- Agent decisions and routing
- System events and errors
- Debugging information

### Conversation Management
- Each conversation has a unique UUID for tracking
- Automatic conversation ending detection
- Seamless transition between topics
- Natural conversation flow

### User Interface
- Clean, intuitive chat interface
- Real-time response generation
- Live log viewing option
- Easy navigation between sections

## Contributors
- Sivan Lissak
- Itay Gefner
- Bonnie Erez

---
*Built with ❤️ using LangChain, OpenAI, and Streamlit*
    """)