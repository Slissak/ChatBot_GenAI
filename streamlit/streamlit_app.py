import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, date
import os
import sys
import time

# Add both project root and app directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app_dir = os.path.join(project_root, 'app')
sys.path.extend([project_root, app_dir])

# Now we can import from app
from app.main import SupervisorAgent

# Set page config
st.set_page_config(
    page_title="Tech app chat bot",
    page_icon="��",
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
    log_path = os.path.join(project_root, "logs", "agent_communications.log")
    
    # Only update log if 2 seconds have passed since last update
    current_time = time.time()
    if current_time - st.session_state.last_log_update >= 2:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()[-50:]  # Get last 50 lines
            log_placeholder.markdown("#### Live Agent Log")
            log_placeholder.code("".join(lines), language="text")
        else:
            log_placeholder.info("Log file not found: agent_communications.log")
        st.session_state.last_log_update = current_time

# Add footer to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("*Built with ❤️ using Streamlit | Powered by AI*")
st.sidebar.markdown("*Contributors: Sivan Lissak, Itay Gefner, Bonnie Erez*")

if page == "main":
    st.header("🏠 Main Chat Area")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the interview process..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from supervisor
        with st.chat_message("assistant"):
            response = st.session_state.supervisor.process_message(prompt)
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Check if conversation should end
            if response.strip() == "[END]":
                # Start new conversation but keep history
                new_id = st.session_state.supervisor.start_new_conversation()
                st.session_state.conversation_id = new_id
                # Add a visual separator for the new conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "---\n*New conversation started*"
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
This application is a multi-agent system designed to simulate and manage job interview conversations. It uses AI agents to provide information, schedule interviews, and determine when to end conversations, all orchestrated by a supervisor agent. The system is built with LangChain, incorporates Retrieval-Augmented Generation (RAG), and integrates with databases and fine-tuned models for efficient and natural interactions.

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

## Application Initialization
The application initializes by first setting up the Supervisor Agent, which then initializes the other three agents (Info, Scheduling, and Exit). For each new conversation, a unique UUID is generated to track and log the interaction uniquely.

## Logging
The system provides extensive logging capabilities. Logs are stored in the `app/logs/` directory, allowing you to track chat turns, agent decisions, and system events for debugging and analysis.

## Streamlit Interface
Users can interact with the chat bot through a Streamlit web application located in `streamlit/streamlit_app.py`. It provides a user-friendly interface for chatting with the bot and includes an option to view live logs for debugging purposes.

## Contributors
- Sivan Lissak
- Itay Gefner
- Bonnie Erez
    """)