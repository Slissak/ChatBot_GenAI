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
    st.write("""
    Welcome to our AI-powered interview assistant!
    
    ### What we do:
    - Connect talented developers with amazing tech opportunities
    - Streamline the interview process with AI assistance
    - Provide personalized career guidance
    
    ### How it works:
    1. Chat with our AI assistant about your interests
    2. Schedule interviews that match your skills
    3. Get real-time feedback and guidance
    
    ### Get Started:
    Simply start chatting in the main section and let our AI guide you!
    """)
