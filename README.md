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

## Setup and Run
Follow these steps to set up and run the application:

1.  **Set up the Python Environment:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate  # On Windows
    
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**
    Create a file named `.env` in the project root. Copy the example below and replace the placeholder values with your actual credentials.
    ```env
    # OpenAI API Configuration
    OPENAI_API_KEY="your_openai_api_key"

    # Database Configuration
    DB_SERVER="your_server_name"
    DB_NAME="your_database_name"
    DB_USER="your_username"
    DB_PASSWORD="your_password"
    DB_PORT="1433" # Default SQL Server port
    DB_DRIVER="ODBC Driver 17 for SQL Server"
    ```

3.  **Set up the Database:**
    - Make sure you have Microsoft SQL Server running and accessible.
    - Create a new database for the application.
    - Run the `app/db_Tech.sql` script in your SQL Server instance to create the `InterviewSlots` table and populate it with initial data.

4.  **Run the Application:**
    Use the Streamlit interface to interact with the bot:
    ```bash
    streamlit run streamlit/streamlit_app.py
    ```

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
The bot's architecture diagram illustrates the structural design of our multi-agent system. This architecture serves as a blueprint for how different agents interact within the system, showing the hierarchical relationship between the supervisor agent and its sub-agents.

![Bot Architecture](app/modules/Bot%20atchitecutre/job_interview_bot_-_multi-agent_system.png)

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