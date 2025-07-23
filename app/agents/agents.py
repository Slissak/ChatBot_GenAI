import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import uuid
import logging
import torch
import lancedb
import platform
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import LanceDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use absolute import
from app.logging_config import (
    log_system_health,
    log_agent_communication,
    log_conversation_event
)
from app.utils.utils import (
    get_db_connection,
    predict_convo_ending
)

from langgraph_supervisor import create_supervisor

class info_agent:
    """Information agent that handles job-related queries using RAG."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.logger = logging.getLogger('info_agent')
        self.logger.info("🚀 Initializing Info Agent")
        
        # Initialize components
        self._init_llm(model_name)
        self._init_vector_store()
        self._init_tools()
        self._load_documents()

    def _init_llm(self, model_name):
        """Initialize the language model."""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7, streaming=True)
        log_system_health("Info Agent LLM", "INITIALIZED", {"model": model_name})

    def _init_vector_store(self):
        """Initialize the vector store for document storage."""
        self.logger.info("🔧 Initializing vector store...")
        db_uri = "tmp/LanceDb"
        table_name = "rag_docs"
        
        try:
            db = lancedb.connect(db_uri)
            if table_name in db.table_names():
                self.logger.info(f"🗑️ Dropping existing table: '{table_name}'")
                db.drop_table(table_name)
                log_system_health("Vector Store Table", "RESET", {"table": table_name})
        except Exception as e:
            self.logger.warning(f"⚠️ Could not drop table '{table_name}': {str(e)}")
            log_system_health("Vector Store Table", "WARNING", {"error": str(e)})
        
        try:
            self.vector_store = LanceDB(
                uri=db_uri,
                table_name=table_name,
                embedding=OpenAIEmbeddings(model="text-embedding-3-small")
            )
            log_system_health("Vector Store", "INITIALIZED", {
                "uri": db_uri, 
                "table": table_name,
                "embedding_model": "text-embedding-3-small"
            })
        except Exception as e:
            self.logger.error(f"❌ Error initializing vector store: {str(e)}")
            log_system_health("Vector Store", "ERROR", {"error": str(e)})
            raise

    def _init_tools(self):
        """Initialize agent tools."""
        self.system_prompt = """
You are a Retrieval-Augmented Generation (RAG) assistant. You expect to receieve a question regarding a job position
on EVERY user question, you MUST first retrieve relevant documents from the knowledge base using the 'retrieve_documents' tool. 
Then, answer the user's question using ONLY the information found in the retrieved documents. 
If no relevant documents are found, reply: 'No relevant information found in the knowledge base.' 
Do NOT use your own knowledge or make up answers. Always cite the retrieved content.
"""
        self.logger.info("📝 System prompt configured")

        @tool
        def retrieve_documents(query: str) -> str:
            """Searches the knowledge base for information about the Python Developer job description."""
            self.logger.info(f"Tool called with query: {query}")
            return self._retrieve_documents(query)

        self.retrieve_documents_tool = retrieve_documents
        self.agent = create_react_agent(
            self.llm,
            tools=[retrieve_documents],
            prompt=self.system_prompt,
            name="info_agent"
        )
        log_system_health("Info Agent", "INITIALIZED", {"tools_count": 1})

    def _retrieve_documents(self, query: str) -> str:
        """Actual retrieval logic for relevant documents from the knowledge base"""
        try:
            self.logger.info(f"🔍 Executing similarity search for query: {query[:100]}...")

            if not hasattr(self, 'vector_store'):
                error_msg = "Error: Vector store not initialized"
                self.logger.error(f"❌ {error_msg}")
                return json.dumps({"error": error_msg})

            # Check document count
            table = self.vector_store._table
            doc_count = table.count_rows()
            self.logger.debug(f"📊 Total documents in vector store: {doc_count}")
            
            if doc_count == 0:
                error_msg = "Error: No documents found in the vector store."
                self.logger.error(f"❌ {error_msg}")
                return json.dumps({"error": error_msg})

            # Perform search
            docs = self.vector_store.similarity_search(query, k=4)
            
            if not docs:
                self.logger.info("No documents found for query")
                return json.dumps([])
            
            # Format results
            formatted_docs = []
            for doc in docs:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0),
                        "chunk": doc.metadata.get("chunk", 0)
                    }
                })
            
            return json.dumps(formatted_docs)
            
        except Exception as e:
            error_msg = f"Error during document retrieval: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            return json.dumps({"error": error_msg})

    def _load_documents(self):
        """Process and add a PDF file to the knowledge base"""
        pdf_path = "app/PythonDeveloperJobDescription.pdf"
        try:
            self.logger.info(f"\n=== Loading PDF: {pdf_path} ===")
            
            if not os.path.exists(pdf_path):
                error_msg = f"Error: PDF file not found at {pdf_path}"
                self.logger.error(f"✗ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Load PDF
            loader = PDFMinerLoader(pdf_path)
            pages = loader.load()
            self.logger.info(f"✓ Successfully loaded {len(pages)} pages")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(pages)
            self.logger.info(f"✓ Split into {len(chunks)} chunks")
            
            # Convert to documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": chunk.metadata.get("page", 0),
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            self.logger.info("Adding documents to vector store...")
            self.vector_store.add_documents(documents)
            self.logger.info(f"✓ Successfully added {len(documents)} documents")
            
            # Verify the knowledge base
            test_query = "What are the Objectives of the job?"
            test_result = self._retrieve_documents(test_query)
            if "error" in test_result.lower():
                raise Exception("Knowledge base verification failed")
            
            log_system_health("Knowledge Base", "ALIVE", {"test_query": test_query})
            self.logger.info("✅ Info Agent Initialization Complete")
            
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            raise

class sched_agent:
    """Scheduling agent that handles interview slot management."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.logger = logging.getLogger('sched_agent')
        self.logger.info("📅 Initializing sched_agent...")
        
        # Initialize components
        self._init_llm(model_name)
        self._init_db_config()
        self._init_tools()

    def _init_llm(self, model_name):
        """Initialize the language model."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True
        )
        log_system_health("Sched Agent LLM", "INITIALIZED", {"model": model_name})

    def _init_db_config(self):
        """Initialize database configuration."""
        self.logger.info("🔍 Checking database environment variables...")
        required_vars = ['DB_DRIVER', 'DB_SERVER', 'DB_NAME', 'DB_PORT', 'DB_USER', 'DB_PASSWORD']
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                self.logger.debug(f"✓ {var} is set")
            else:
                self.logger.warning(f"⚠️ {var} is not set")
                missing_vars.append(var)
        
        if missing_vars:
            log_system_health("Database Environment", "WARNING", {"missing_vars": missing_vars})
        else:
            log_system_health("Database Environment", "ALIVE", {"all_vars_set": True})
        
        self.db_config = {
            'driver': os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
            'server': os.getenv('DB_SERVER'),
            'database': os.getenv('DB_NAME'),
            'port': os.getenv('DB_PORT'),
            'uid': os.getenv('DB_USER'),
            'pwd': os.getenv('DB_PASSWORD'),
            'Connection Timeout': '30'
        }
        self.logger.info("✓ Database configuration loaded")

    def _init_tools(self):
        """Initialize agent tools."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.system_prompt = f"""
        You are a scheduling assistant for technical interviews. The current date is {today}.

        Your task is to provide interview slots. You have access to a tool called 'query_available_slots' that you MUST use to check availability.

        The tool takes these parameters:
        - date (required): in YYYY-MM-DD format
        - time_frame (optional): 'morning', 'afternoon', or 'evening'
        - position (optional): job position (defaults to 'Python Dev' if not specified)

        Example usage:
        User: "I want to schedule a Python Dev interview for 2025-03-01 in the morning"
        You should:
        1. Extract date="2025-03-01", time_frame="morning", position="Python Dev"
        2. Call query_available_slots with these parameters
        3. Format the response for the user

        Always use the tool to check availability. Do not make up answers. If the user input is ambiguous, ask for clarification.
        """
        self.logger.info("✓ System prompt configured")

        @tool
        def query_available_slots(date: str, time_frame: str = None, position: str = None) -> str:
            """Query the database for available interview slots."""
            self.logger.info(f"\nQuerying available slots with parameters:")
            self.logger.info(f"Date: {date}")
            self.logger.info(f"Time frame: {time_frame}")
            self.logger.info(f"Position: {position}")
            
            try:
                # Validate date format
                try:
                    datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    return json.dumps({
                        "error": "Invalid date format. Please use YYYY-MM-DD format."
                    })
                
                # Set default position
                if not position:
                    position = "Python Dev"
                
                query = """
                SELECT ScheduleID, [date], [time], position, available
                FROM dbo.Schedule
                WHERE [date] = ? AND position = ? AND available = 0
                """
                params = [date, position]
                
                if time_frame:
                    if time_frame == 'morning':
                        query += " AND [time] BETWEEN '09:00' AND '12:00'"
                    elif time_frame == 'afternoon':
                        query += " AND [time] BETWEEN '12:00' AND '17:00'"
                    elif time_frame == 'evening':
                        query += " AND [time] BETWEEN '17:00' AND '20:00'"
                    else:
                        return json.dumps({
                            "error": "Invalid time frame. Use 'morning', 'afternoon', or 'evening'."
                        })
                
                query += " ORDER BY [time]"
                
                try:
                    with get_db_connection(self.db_config) as conn:
                        cursor = conn.cursor()
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
                        
                        available_slots = []
                        for row in rows:
                            slot = {
                                "schedule_id": row.ScheduleID,
                                "date": row.date.strftime('%Y-%m-%d'),
                                "time": row.time.strftime('%H:%M'),
                                "position": row.position
                            }
                            available_slots.append(slot)
                        
                        if not available_slots:
                            # Check alternative dates
                            alt_dates = []
                            for i in range(1, 4):
                                alt_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                                cursor.execute(query, [alt_date, position])
                                alt_rows = cursor.fetchall()
                                if alt_rows:
                                    alt_dates.append(alt_date)
                            
                            return json.dumps({
                                "date": date,
                                "time_frame": time_frame,
                                "position": position,
                                "available_slots": [],
                                "suggested_dates": alt_dates,
                                "message": "No available slots found for the requested date. Suggested alternative dates provided."
                            })
                        
                        return json.dumps({
                            "date": date,
                            "time_frame": time_frame,
                            "position": position,
                            "available_slots": available_slots,
                            "total_slots": len(available_slots)
                        })
                        
                except Exception as db_error:
                    error_msg = f"Database error: {str(db_error)}"
                    self.logger.error(f"✗ {error_msg}")
                    return json.dumps({"error": error_msg})
                    
            except Exception as e:
                error_msg = f"Error querying available slots: {str(e)}"
                self.logger.error(f"✗ {error_msg}")
                return json.dumps({"error": error_msg})

        # Create the agent with the tool
        self.agent = create_react_agent(
            self.llm,
            tools=[query_available_slots],
            prompt=self.system_prompt,
            name="sched_agent"
        )
        self.logger.info("✓ Agent created successfully")

    def chat(self, user_message: str) -> str:
        """Process a chat message and return the response."""
        try:
            messages = [HumanMessage(content=user_message)]
            response = self.agent.invoke({"messages": messages})
            
            if isinstance(response, dict) and 'messages' in response:
                return response['messages'][-1].content
            else:
                return "Error: No response from agent"
                
        except Exception as e:
            self.logger.error(f"✗ Error in chat: {str(e)}")
            return f"Error: {str(e)}"

class SmartExitAgent:
    """Exit agent that determines when to end conversations."""
    def __init__(self, model_name: str = "ft:gpt-4.1-2025-04-14:personal::BnnSEmUJ"):
        self.logger = logging.getLogger('smart_exit_agent')
        self.logger.info(f"🚦 Initializing SmartExitAgent...")
        
        # Initialize components
        self._init_llm(model_name)
        self._init_conversation_model()
        self._init_tools()
        
        # Cache for conversation analysis
        self.analysis_cache = {}

    def _init_llm(self, model_name):
        """Initialize the language model."""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.finetuned_model_name = model_name
        self.logger.info(f"OpenAI fine-tuned model set: {model_name}")
        log_system_health("Exit Agent LLM", "INITIALIZED", {"model": model_name})

    def _init_conversation_model(self):
        """Initialize the BERT conversation model."""
        self.conversation_model = None
        self.conversation_tokenizer = None
        self.device = self._setup_device()
        log_system_health("BERT Model", "INITIALIZED", {"device": str(self.device)})

    def _setup_device(self):
        """Setup device with Apple Silicon optimization."""
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"System: {platform.platform()}")
        
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("🚀 Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using NVIDIA CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("⚠️ Using CPU only")
        
        return device

    def _init_tools(self):
        """Initialize agent tools."""
        self.system_exit_prompt = """You are an exit advisor. Your mission is to decide if the conversation with a job candidate should end, 
        ALWAYS use the tool 'analyze_conversation_ending' to determine if the conversation should end, the tool will return only two options,
        "END" - indicating that the conversation should end, "CONTINUE" - indicating that the conversation should continue.
        forward the result of the tool back to the supervisor agent."""

        @tool
        def analyze_conversation_ending(conversation_history: str) -> str:
            """Analyze if the conversation should end using the OpenAI fine-tuned model"""
            try:
                # Use OpenAI fine-tuned model for prediction
                openai_result = self._predict_with_finetuned_model(conversation_history)
                self.logger.info(f"OpenAI fine-tuned model prediction: {openai_result}")
                
                # Extract conversation features
                features = self._extract_conversation_features(conversation_history)
                
                # Combine predictions and features
                analysis = {
                    "openai_prediction": openai_result.get("prediction", "UNKNOWN"),
                    "openai_confidence": openai_result.get("confidence", None),
                    "conversation_features": features,
                    "recommendation": self._make_final_decision(openai_result, features)
                }
                
                self.logger.info(f"Conversation analysis: {analysis}")
                return json.dumps(analysis)
                
            except Exception as e:
                self.logger.error(f"Error in conversation analysis: {e}")
                return json.dumps({"error": str(e), "fallback": "CONTINUE"})

        # Create agent with tools
        self.agent = create_react_agent(
            self.llm,
            tools=[analyze_conversation_ending],
            prompt=self.system_exit_prompt,
            name="smart_exit_agent"
        )
        self.logger.info("✓ Agent created successfully")

    def _predict_with_finetuned_model(self, conversation_text: str) -> dict:
        """Use OpenAI fine-tuned model to predict conversation ending."""
        try:
            self.logger.info(f"Using OpenAI fine-tuned model ({self.finetuned_model_name}) for prediction")
            
            prompt = f"""
You are a classifier. Given the following conversation, respond with a JSON object with two fields: 'prediction' (either 'END' or 'NOT_END') and 'confidence' (a float between 0 and 1 indicating your confidence in the prediction).

Conversation:
{conversation_text}

Respond ONLY with the JSON object.
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Try to extract JSON from the response
            import re
            import ast
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                try:
                    result = ast.literal_eval(match.group(0))
                    if "prediction" in result:
                        return result
                except Exception as e:
                    self.logger.warning(f"Could not parse OpenAI model response as JSON: {e}")
            
            # Fallback to NOT_END if no valid JSON
            return {"prediction": "NOT_END", "confidence": 0.5}
            
        except Exception as e:
            self.logger.error(f"Error using OpenAI fine-tuned model: {e}")
            return {"prediction": "UNKNOWN", "confidence": 0.0}

    def _extract_conversation_features(self, conversation: str) -> dict:
        """Extract rule-based features for hybrid decision making."""
        features = {
            "has_goodbye_words": any(word in conversation.lower() 
                                   for word in ["bye", "goodbye", "farewell", "see you"]),
            "has_thank_you": "thank" in conversation.lower(),
            "mentions_scheduling": any(word in conversation.lower() 
                                     for word in ["schedule", "interview", "appointment"]),
            "expresses_disinterest": any(phrase in conversation.lower() 
                                       for phrase in ["not interested", "found a job", "no thanks"]),
            "conversation_length": len(conversation.split()),
            "question_marks": conversation.count("?"),
            "exclamation_marks": conversation.count("!")
        }
        return features

    def _make_final_decision(self, model_result: dict, features: dict) -> str:
        """Combine OpenAI model with rule-based logic for final decision."""
        # Prefer OpenAI model if confident
        if model_result.get("confidence", 0) > 0.8:
            self.logger.info(f"Final decision based on OpenAI model: {model_result['prediction']}")
            return model_result["prediction"]
            
        # Rule-based overrides
        if features["expresses_disinterest"] or features["mentions_scheduling"]:
            self.logger.info("Rule-based override: END due to disinterest or scheduling mention")
            return "END"
            
        # Multiple ending signals
        ending_signals = sum([
            features["has_goodbye_words"],
            features["has_thank_you"],
            model_result.get("prediction", "NOT_END") == "END"
        ])
        
        if ending_signals >= 2:
            self.logger.info("Rule-based override: END due to multiple ending signals")
            return "END"
            
        self.logger.info("Final decision: NOT_END")
        return "NOT_END"

class SupervisorAgent:
    """Main supervisor agent that orchestrates other agents."""
    def __init__(self, model_name: str = "gpt-4o", conversation_id: str = None):
        self.logger = logging.getLogger('supervisor_agent')
        self.logger.info("🎼 Initializing SupervisorAgent...")
        
        # Initialize core components
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.memory = MemorySaver()
        self.last_exit_decision = None
        
        # Initialize sub-agents
        self._init_llm(model_name)
        self._init_sub_agents()
        self._init_workflow()
        
        log_conversation_event(self.conversation_id, "started", {"supervisor_initialized": True})

    def _init_llm(self, model_name):
        """Initialize the language model."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True
        )
        log_system_health("Supervisor LLM", "INITIALIZED", {"model": model_name})

    def _init_sub_agents(self):
        """Initialize all sub-agents."""
        self.info_agent = info_agent()
        self.exit_agent = SmartExitAgent()
        self.sched_agent = sched_agent()
        
        log_system_health("All Sub-agents", "INITIALIZED", {
            "agents": ["info_agent", "smart_exit_agent", "sched_agent"]
        })

    def _init_workflow(self):
        """Initialize the supervisor workflow."""
        # Create date-aware system prompt
        self.system_supervisor_prompt = self._create_date_aware_prompt()
        
        # Create the supervisor workflow
        workflow = create_supervisor(
            agents=[
                self.info_agent.agent,
                self.exit_agent.agent,
                self.sched_agent.agent
            ],
            model=self.llm,
            prompt=self.system_supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history"
        )
        
        # Compile the workflow
        self.workflow = workflow.compile(
            name="supervisor_workflow",
            checkpointer=self.memory
        )
        log_system_health("Supervisor Workflow", "INITIALIZED", {"name": "supervisor_workflow"})

    def _create_date_aware_prompt(self) -> str:
        """Create supervisor prompt with current date information."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_year = datetime.now().year
        current_day = datetime.now().strftime('%A')
        
        return f"""
You are a professional and friendly AI assistant for Tech's Python Developer position. Your role is to provide a warm, engaging experience while helping candidates learn about the position and schedule interviews.

CURRENT DATE INFORMATION:
- Today's date: {current_date}
- Day of week: {current_day}
- Current year: {current_year}

CONVERSATION FLOW:
1. Initial Greeting:
   - Start with a warm, professional greeting
   - Introduce yourself as Tech's Python Developer position assistant
   - Ask if they're interested in learning more about the Python Developer role or would like to schedule an interview
   Example: "Hello! 👋 I'm Tech's Python Developer position assistant. I'd be happy to tell you more about this exciting opportunity or help you schedule an interview. Would you like to learn more about the role?"

2. Based on their response:
   - If they want to learn more: Use info_agent to provide accurate details about the position
   - If they want to schedule: Use sched_agent to find available interview slots
   - If they're not interested: Acknowledge politely and use exit_agent to determine next steps
   - If they indicate they want to end the conversation: ALWAYS use exit_agent to confirm

3. Throughout the conversation:
   - For job-related questions: ALWAYS use info_agent to provide accurate information
   - For scheduling: Use sched_agent to find and book interview slots
   - For ANY potential conversation ending: ALWAYS use exit_agent to determine if the conversation should conclude

AGENT USAGE RULES:
- info_agent: 
  * Use for ALL job-related questions. Never answer job questions yourself.
  * ALWAYS use the info_agent's response, summarize it and then relay it to the user
  * Only add a brief follow-up like "Let me know if you need any clarification or have other questions"

- sched_agent: 
  * Use for checking interview availability and scheduling
  * After scheduling, ALWAYS use exit_agent to check if conversation should end

- exit_agent: 
  * MUST use for ANY potential conversation ending signals:
    - User says goodbye, bye, thanks, etc.
    - User expresses disinterest
    - User has scheduled an interview
    - User indicates they're done
  * ALWAYS use exit_agent when:
    - After successful interview scheduling
    - When user expresses disinterest
    - When user says goodbye or indicates end
    - When user has been inactive
  * NEVER end the conversation without consulting exit_agent
  * When exit_agent returns END, end the conversation gracefully

KEY BEHAVIORS:
- Be warm and professional at all times
- Prioritize scheduling interviews when appropriate
- Answer questions thoroughly using info_agent
- NEVER modify or replace info_agent's responses
- Confirm interview details after scheduling
- Always ask if they need anything else after providing information
- Check with exit_agent for ANY potential conversation ending
- End conversations gracefully when exit_agent indicates END

Remember: Your goal is to either schedule an interview or provide helpful information about the position, while maintaining a professional and engaging conversation. ALWAYS consult exit_agent before ending any conversation, regardless of the context or timing."""

    def process_message(self, message: str) -> str:
        """Process a user message through the supervisor workflow."""
        try:
            self.logger.info(f"💬 Processing user message: {message[:100]}...")
            log_conversation_event(self.conversation_id, "message", {
                "direction": "user_to_supervisor",
                "message_length": len(message)
            })
            
            # Create initial message
            messages = [HumanMessage(content=message)]
            
            # Invoke workflow with conversation tracking
            config = {"configurable": {"thread_id": self.conversation_id}}
            response = self.workflow.invoke({"messages": messages}, config=config)
            
            # Process intermediate steps and agent decisions
            if 'intermediate_steps' in response:
                self.logger.info("🎯 === AGENT SELECTION AND MESSAGE ROUTING ===")
                for step in response['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) > 1:
                        action, result = step
                        if hasattr(action, 'name'):
                            # Track exit agent decisions
                            if action.name == "smart_exit_agent":
                                try:
                                    if isinstance(result, dict):
                                        analysis = json.loads(result) if isinstance(result, str) else result
                                        self.last_exit_decision = analysis.get('recommendation') or analysis.get('prediction')
                                    elif isinstance(result, str):
                                        self.last_exit_decision = result.strip('[]')
                                except:
                                    self.last_exit_decision = None
                                self.logger.info(f"🚪 Exit agent decision: {self.last_exit_decision}")
                            
                            # Log agent communications
                            log_agent_communication(
                                'agent_step',  # event_type
                                self.conversation_id,
                                action.name,
                                'to_agent',
                                str(action.input)
                            )
                            log_agent_communication(
                                'agent_response',  # event_type
                                self.conversation_id,
                                action.name,
                                'from_agent',
                                str(result)
                            )
            
            # Extract and format final response
            if 'messages' in response and response['messages']:
                final_response = response['messages'][-1].content
                
                messages_length = len(response['messages'])
                for i in range(messages_length - 1, -1, -1):  # Start from last index to 0
                    message = response['messages'][i]
                    # Check if the content contains the specific pattern
                    if isinstance(message.content, str) and "openai_prediction" in message.content:
                        prediction_data = json.loads(message.content)
                        self.last_exit_decision = prediction_data.get('recommendation')
                        self.logger.info(f"🚪 Exit agent decision: {self.last_exit_decision}")
                        break

                # Add END marker if exit agent decided to end
                if self.last_exit_decision == 'END':
                    final_response = f"{final_response}\n[END]"
                    self.logger.info("🚪 Adding [END] marker to response")
                
                # Log response
                if len(final_response) <= 300:
                    self.logger.info(f"📤 Final response: {final_response}")
                else:
                    self.logger.info(f"📤 Final response: {final_response[:300]}... [TRUNCATED]")
                
                log_conversation_event(self.conversation_id, "message", {
                    "direction": "supervisor_to_user",
                    "response_length": len(final_response)
                })
                
                return final_response
            else:
                error_msg = "Error: No response from supervisor"
                self.logger.error(f"❌ {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Error in supervisor processing: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            log_system_health("Supervisor Processing", "ERROR", {"error": str(e)})
            return f"Error: {str(e)}"
    

    def start_new_conversation(self) -> str:
        """Start a new conversation session."""
        # Log end of current conversation
        old_id = self.conversation_id
        log_conversation_event(old_id, "ended", {"reason": "new_conversation_started"})
        
        # Generate new conversation ID
        self.conversation_id = str(uuid.uuid4())
        log_conversation_event(self.conversation_id, "started", {"previous_id": old_id})
        
        # Refresh date context
        self.refresh_date_context()
        
        self.logger.info(f"🆕 Starting new conversation session: {self.conversation_id}")
        return self.conversation_id

    def refresh_date_context(self):
        """Refresh the supervisor prompt with current date."""
        self.system_supervisor_prompt = self._create_date_aware_prompt()
        
        # Recreate workflow with updated prompt
        workflow = create_supervisor(
            agents=[
                self.info_agent.agent,
                self.exit_agent.agent,
                self.sched_agent.agent
            ],
            model=self.llm,
            prompt=self.system_supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history"
        )
        
        self.workflow = workflow.compile(
            name="supervisor_workflow",
            checkpointer=self.memory
        )
        
        self.logger.info(f"🗓️ Refreshed date context - Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}") 