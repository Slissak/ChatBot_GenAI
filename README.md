# Job Interview Bot

## Contributors
- Sivan Lissak
- Itay Gefner
- Bonnie Erez

## Overview
This application is a multi-agent system designed to simulate and manage job interview conversations. It uses AI agents to provide information, schedule interviews, and determine when to end conversations, all orchestrated by a supervisor agent. The system is built with LangChain, incorporates Retrieval-Augmented Generation (RAG), and integrates with databases and fine-tuned models for efficient and natural interactions.



## Bot Architecture
The bot's architecture is a multi-agent system where a supervisor agent coordinates between specialized agents to handle different aspects of the job interview process.

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

## Setup and Installation
To run this application, follow these steps:

1. **Database Setup:**
   - Create a Microsoft SQL Server database.
   - Run the SQL script located at `app/db_Tech.sql` to set up the necessary tables and data.

2. **Environment Configuration:**
   - Create a `.env` file in the project root.
   - Add your database connection details (e.g., DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT, DB_DRIVER).
   - Add your OpenAI API key: `OPENAI_API_KEY=your_openai_api_key`.

3. **Install Dependencies:**
   - Run `pip install -r requirements.txt` from the project root.

4. **Run the Application:**
   - For the Streamlit interface: `streamlit run streamlit/streamlit_app.py`
   - For console mode: `python app/main.py`

   
## Contributors
- Sivan Lissak
- Itay Gefner
- Bonnie Erez 