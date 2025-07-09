import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from typing import Dict, Any, List, Optional
import json
import uuid
from rich import print
from rich.pretty import Pretty, pprint
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_core.tools import tool
from rich.panel import Panel
from rich import print_json
from rich.console import Console, Group
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pyodbc
from datetime import datetime, date, timedelta
from pydantic import BaseModel, Field
import lancedb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
import platform
import psutil

# Import logging configuration
from app.logging_config import (
    setup_logging, 
    log_agent_communication, 
    log_system_health, 
    log_conversation_event
)

# Load environment variables
load_dotenv()

# Initialize comprehensive logging
setup_logging(log_level=logging.DEBUG)
# GITHUB_TOKEN = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
# OAI_TOKEN = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-4o") #"gpt-4.1-nano")
# Define system prompt for our agentd and orchestrator


class info_agent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.logger = logging.getLogger('info_agent')
        self.logger.info("🚀 Initializing Info Agent")
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            streaming=True
        )
        log_system_health("Info Agent LLM", "INITIALIZED", {"model": model_name})
        
        # Set up vector store with LanceDB
        self.logger.info("🔧 Initializing vector store...")
        db_uri = "tmp/LanceDb"
        table_name = "rag_docs"
        
        # Connect to LanceDB and drop the table if it exists to ensure a fresh start
        try:
            db = lancedb.connect(db_uri)
            if table_name in db.table_names():
                self.logger.info(f"🗑️ Dropping existing table: '{table_name}'")
                db.drop_table(table_name)
                log_system_health("Vector Store Table", "RESET", {"table": table_name})
        except Exception as e:
            # This is not a critical error, the process can continue.
            self.logger.warning(f"⚠️ Could not drop table '{table_name}': {str(e)}")
            log_system_health("Vector Store Table", "WARNING", {"error": str(e)})
        
        # create new table and choosing OpenAI embedding model
        try:
            self.vector_store = LanceDB(
                uri=db_uri,
                table_name=table_name,
                embedding=OpenAIEmbeddings(
                    model="text-embedding-3-small"
                ),
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
        
        # Set up memory for conversation history
        # self.memory = None #MemorySaver()
        
        # Context7-style system prompt for strict RAG
        self.system_prompt = """
You are a Retrieval-Augmented Generation (RAG) assistant. You expect to receieve a question regarding a job position
on EVERY user question, you MUST first retrieve relevant documents from the knowledge base using the 'retrieve_documents' tool. 
Then, answer the user's question using ONLY the information found in the retrieved documents. 
If no relevant documents are found, reply: 'No relevant information found in the knowledge base.' 
Do NOT use your own knowledge or make up answers. Always cite the retrieved content.
"""
        self.logger.info("📝 System prompt configured")
        
        # Create the retrieve_documents tool properly
        @tool
        def retrieve_documents(query: str) -> str:
            """Searches the knowledge base for information about the Python Developer job description."""
            self.logger.info("\n=== Info Agent Tool Usage ===")
            self.logger.info(f"Tool called with query: {query}")
            self.logger.info("Retrieving documents from knowledge base...")
            
            try:
                result = self._retrieve_documents(query)
                self.logger.info("=== Tool Execution Complete ===\n")
                return result
            except Exception as e:
                error_msg = f"Error in retrieve_documents tool: {str(e)}"
                self.logger.error(f"✗ {error_msg}")
                return json.dumps({"error": error_msg})
        
        # Store the tool function for access
        self.retrieve_documents_tool = retrieve_documents
        
        # Create the LangGraph agent with RAG capabilities
        self.agent = create_react_agent(
            self.llm,
            tools=[retrieve_documents],  
            prompt=self.system_prompt,
            # checkpointer=self.memory,
            name="info_agent"
        )
        log_system_health("Info Agent", "INITIALIZED", {"tools_count": 1})

        # Load the PDF and verify the knowledge base
        self.logger.info("📄 Loading PDF into knowledge base...")
        result = self.add_pdf("app/PythonDeveloperJobDescription.pdf")
        if "Error" in result:
            self.logger.error(f"❌ {result}")
            log_system_health("PDF Loading", "ERROR", {"result": result})
            raise Exception(f"Failed to load PDF: {result}")
        
        log_system_health("PDF Loading", "ALIVE", {"file": "PythonDeveloperJobDescription.pdf"})
        
        # Test retrieval to verify system
        test_retrieval_result = self._retrieve_documents("What programming languages are needed for this role??")
        self.logger.debug(f"🧪 Manual retrieval test result: {test_retrieval_result[:200]}...")
        
        # Verify the knowledge base is working
        self.logger.info("🔍 Verifying knowledge base...")
        test_query = "What are the Objectives of the job?"
        test_result = self._retrieve_documents(test_query)
        if "error" in test_result.lower():
            self.logger.error(f"❌ Knowledge base verification failed: {test_result}")
            log_system_health("Knowledge Base", "ERROR", {"test_query": test_query})
            raise Exception("Knowledge base verification failed")
        
        log_system_health("Knowledge Base", "ALIVE", {"test_query": test_query})
        self.logger.info("✅ Info Agent Initialization Complete")

    def _retrieve_documents(self, query: str) -> str:
        """Actual retrieval logic for relevant documents from the knowledge base"""
        try:
            self.logger.info(f"🔍 Executing similarity search for query: {query[:100]}...")

            # Verify vector store is initialized
            if not hasattr(self, 'vector_store'):
                error_msg = "Error: Vector store not initialized"
                self.logger.error(f"❌ {error_msg}")
                log_system_health("Vector Store", "ERROR", {"error": error_msg})
                return json.dumps({"error": error_msg})

            # Check if there are any documents in the vector store
            try:
                table = self.vector_store._table
                doc_count = table.count_rows()
                self.logger.debug(f"📊 Total documents in vector store: {doc_count}")
                if doc_count == 0:
                    error_msg = "Error: No documents found in the vector store."
                    self.logger.error(f"❌ {error_msg}")
                    log_system_health("Vector Store Documents", "ERROR", {"count": 0})
                    return json.dumps({"error": error_msg})
                log_system_health("Vector Store Documents", "ALIVE", {"count": doc_count})
            except Exception as e:
                error_msg = f"Error counting documents in vector store: {e}"
                self.logger.error(f"❌ {error_msg}")
                log_system_health("Vector Store", "ERROR", {"error": str(e)})
                return json.dumps({"error": error_msg})

            # Perform the search
            try:
                docs = self.vector_store.similarity_search(
                    query,
                    k=4,
                    search_type="similarity"  # Changed from "hybrid"
                )
            except Exception as search_error:
                self.logger.error(f"✗ Error during similarity search: {str(search_error)}")
                return json.dumps({"error": f"Search error: {str(search_error)}"})
            
            if not docs:
                self.logger.info("No documents found for query")
                return json.dumps([])
            
            # Format the results
            formatted_docs = []
            for doc in docs:
                try:
                    formatted_docs.append({
                        "content": doc.page_content,
                        "metadata": {
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", 0),
                            "chunk": doc.metadata.get("chunk", 0)
                        }
                    })
                except Exception as format_error:
                    self.logger.error(f"✗ Error formatting document: {str(format_error)}")
                    continue
            
            if not formatted_docs:
                self.logger.info("No valid documents after formatting")
                return json.dumps([])
            
            return json.dumps(formatted_docs)
            
        except Exception as e:
            error_msg = f"Error during document retrieval: {str(e)}"
            print(f"✗ {error_msg}")
            return json.dumps({"error": error_msg})

    def add_pdf(self, pdf_path: str) -> str:
        """Process and add a PDF file to the knowledge base"""
        try:
            print(f"\n=== Loading PDF: {pdf_path} ===")
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                error_msg = f"Error: PDF file not found at {pdf_path}"
                print(f"✗ {error_msg}")
                return error_msg
            
            # Load the PDF
            loader = PDFMinerLoader(pdf_path)
            pages = loader.load()
            print(f"✓ Successfully loaded {len(pages)} pages")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(pages)
            print(f"✓ Split into {len(chunks)} chunks")
            
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
            print("\nAdding documents to vector store...")
            try:
                self.vector_store.add_documents(documents)
                print(f"✓ Successfully added {len(documents)} documents")
            except Exception as e:
                error_msg = f"Error adding documents to vector store: {str(e)}"
                print(f"✗ {error_msg}")
                return error_msg
            
            return "PDF loaded and processed successfully"
            
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

class SmartExitAgent:
    def __init__(self, model_name: str = "ft:gpt-4.1-2025-04-14:personal::BnnSEmUJ"):
        self.logger = logging.getLogger('smart_exit_agent')
        self.logger.info(f"🚦 Initializing SmartExitAgent with OpenAI fine-tuned model: {model_name}")
        # Initialize LLM for reasoning (OpenAI fine-tuned model)
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.finetuned_model_name = model_name
        self.logger.info(f"OpenAI fine-tuned model set: {model_name}")
        # Load your trained conversation ending model (BERT, backup only)
        self.conversation_model = None
        self.conversation_tokenizer = None
        self.device = self._setup_device()
        # Load the trained model (BERT, backup only)
        # self._load_conversation_model()
        # Preload model on initialization to avoid delays (BERT, backup only)
        # self._warmup_model()
        # exit advisor prompt
        self.system_exit_prompt = """You are an exit advisor. Your mission is to decide if the conversation with a job candidate should end, 
        ALWAYS use the tool 'analyze_conversation_ending' to determine if the conversation should end, the tool will return only two options,
        "END" - indicating that the conversation should end, "CONTINUE" - indicating that the conversation should continue.
        forward the rersult of the tool back to the supervisor agent."""

        # Create the analyze_conversation_ending tool properly
        @tool
        def analyze_conversation_ending(conversation_history: str) -> str:
            """Analyze if the conversation should end using the OpenAI fine-tuned model"""
            try:
                # Use only the OpenAI fine-tuned model for prediction
                openai_result = self._predict_with_finetuned_model(conversation_history)
                self.logger.info(f"OpenAI fine-tuned model prediction: {openai_result}")
                features = self._extract_conversation_features(conversation_history)
                analysis = {
                    "openai_prediction": openai_result.get("prediction", "UNKNOWN"),
                    "openai_confidence": openai_result.get("confidence", None),
                    "conversation_features": features,
                    "recommendation": self._make_final_decision(openai_result, features)
                }
                self.logger.info(f"Conversation analysis (OpenAI only): {analysis}")
                return json.dumps(analysis)
            except Exception as e:
                self.logger.error(f"Error in conversation analysis: {e}")
                return json.dumps({"error": str(e), "fallback": "continue"})
        # Store the tool function for access
        self.analyze_conversation_ending_tool = analyze_conversation_ending
        # Create agent with enhanced tools
        self.agent = create_react_agent(
            self.llm,
            tools=[analyze_conversation_ending],
            prompt=self.system_exit_prompt,
            name="smart_exit_agent"
        )
        # Cache for conversation analysis (avoid re-analyzing same text)
        self.analysis_cache = {}

    def _setup_device(self):
        """Setup device with Apple Silicon M4 optimization"""
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"System: {platform.platform()}")
        
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("🚀 Using Apple Silicon MPS (Metal Performance Shaders)")
            logging.info("Your trained model will run on M4 GPU!")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Using NVIDIA CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logging.info("⚠️  Using CPU only - model inference will be slower")
        
        # Memory info
        if hasattr(psutil, 'virtual_memory'):
            memory = psutil.virtual_memory()
            logging.info(f"Available RAM: {memory.total // (1024**3)}GB")
        
        return device

    def _load_conversation_model(self):
        """Load the trained BERT conversation ending model"""
        try:
            model_path = "./final_model"  # Your trained model path
            logging.info(f"🤖 Loading BERT conversation ending model from {model_path}")
            
            self.conversation_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.conversation_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.conversation_model.to(self.device)
            self.conversation_model.eval()
            
            log_system_health("BERT Model", "INITIALIZED", {
                "model_path": model_path,
                "device": str(self.device)
            })
        except Exception as e:
            logging.warning(f"⚠️ Could not load BERT conversation model: {e}")
            logging.warning("🔄 Falling back to rule-based detection")
            log_system_health("BERT Model", "ERROR", {
                "error": str(e),
                "fallback": "rule-based"
            })
            self.conversation_model = None

    def _warmup_model(self):
        """Warm up the model with a dummy prediction"""
        if self.conversation_model:
            dummy_text = "Hello, how are you today?"
            self._predict_conversation_ending(dummy_text)
            log_system_health("BERT Model", "ALIVE", {"warmed_up": True})

    def _predict_conversation_ending(self, conversation_text: str) -> dict:
        """Use trained BERT model to predict conversation ending"""
        if self.conversation_model is None:
            return {"prediction": "UNKNOWN", "confidence": 0.5}
        
        # Use the same predict_convo_ending function from your training script
        return predict_convo_ending(
            conversation_text, 
            self.conversation_model, 
            self.conversation_tokenizer, 
            self.device
        )

    def _predict_with_finetuned_model(self, conversation_text: str) -> dict:
        """Use OpenAI fine-tuned model to predict conversation ending"""
        try:
            self.logger.info(f"Using OpenAI fine-tuned model ({self.finetuned_model_name}) for prediction.")
            # Use the LLM to get a prediction (simulate classification)
            # Prompt engineering: ask for END/NOT_END and confidence
            prompt = f"""
You are a classifier. Given the following conversation, respond with a JSON object with two fields: 'prediction' (either 'END' or 'NOT_END') and 'confidence' (a float between 0 and 1 indicating your confidence in the prediction).\n\nConversation:\n{conversation_text}\n\nRespond ONLY with the JSON object.\n"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import re
            import ast
            # Try to extract JSON from the response
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                try:
                    result = ast.literal_eval(match.group(0))
                    if "prediction" in result:
                        return result
                except Exception as e:
                    self.logger.warning(f"Could not parse OpenAI model response as JSON: {e}")
            # Fallback: if no valid JSON, default to NOT_END
            return {"prediction": "NOT_END", "confidence": 0.5}
        except Exception as e:
            self.logger.error(f"Error using OpenAI fine-tuned model: {e}")
            return {"prediction": "UNKNOWN", "confidence": 0.0}

    def _extract_conversation_features(self, conversation: str) -> dict:
        """Extract rule-based features for hybrid decision making"""
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
        """Combine OpenAI model with rule-based logic for final decision"""
        # Prefer OpenAI model if available and confident
        if model_result.get("confidence", 0) > 0.8:
            self.logger.info(f"Final decision based on OpenAI model: {model_result['prediction']}")
            return model_result["prediction"]
        # Rule-based overrides for specific cases
        if features["expresses_disinterest"] or features["mentions_scheduling"]:
            self.logger.info("Rule-based override: END due to disinterest or scheduling mention.")
            return "END"
        # Multiple ending signals
        ending_signals = sum([
            features["has_goodbye_words"],
            features["has_thank_you"],
            model_result.get("prediction", "NOT_END") == "END"
        ])
        if ending_signals >= 2:
            self.logger.info("Rule-based override: END due to multiple ending signals.")
            return "END"
        self.logger.info("Final decision: NOT_END")
        return "NOT_END"

class sched_agent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.logger = logging.getLogger('sched_agent')
        self.logger.info("📅 Initializing sched_agent...")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True
        )
        log_system_health("Sched Agent LLM", "INITIALIZED", {"model": model_name})
        
        # Debug environment variables
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
            'Connection Timeout': '30'  # Add connection timeout
        }
        self.logger.info("✓ Database configuration loaded")
        
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
        
        # Create the tool as a standalone function
        @tool
        def query_available_slots(date: str, time_frame: str = None, position: str = None) -> str:
            """Query the database for available interview slots.
            
            Args:
                date (str): The date to check for available slots in YYYY-MM-DD format
                time_frame (str, optional): The time frame to check (morning, afternoon, or evening)
                position (str, optional): The position to check availability for. Defaults to None
                
            Returns:
                str: JSON string containing available slots or error message
            """
            print(f"\nQuerying available slots with parameters:")
            print(f"Date: {date}")
            print(f"Time frame: {time_frame}")
            print(f"Position: {position}")
            
            try:
                # Validate date format
                try:
                    datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    print("✗ Invalid date format")
                    return json.dumps({
                        "error": "Invalid date format. Please use YYYY-MM-DD format."
                    })
                
                # Set default position if none provided
                if not position:
                    position = "Python Dev"
                    print(f"Using default position: {position}")
                
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
                        print(f"✗ Invalid time frame: {time_frame}")
                        return json.dumps({
                            "error": "Invalid time frame. Use 'morning', 'afternoon', or 'evening'."
                        })
                
                query += " ORDER BY [time]"
                print("\nExecuting database query:")
                print(f"Query: {query}")
                print(f"Parameters: {params}")
                
                try:
                    with self._get_db_connection() as conn:
                        cursor = conn.cursor()
                        print("\nExecuting cursor...")
                        cursor.execute(query, params)
                        print("Query executed successfully")
                        
                        rows = cursor.fetchall()
                        print(f"Number of rows returned: {len(rows)}")
                        
                        available_slots = []
                        for row in rows:
                            slot = {
                                "schedule_id": row.ScheduleID,
                                "date": row.date.strftime('%Y-%m-%d'),
                                "time": row.time.strftime('%H:%M'),
                                "position": row.position
                            }
                            available_slots.append(slot)
                            print(f"Found slot: {slot}")
                        
                        print(f"\nTotal available slots found: {len(available_slots)}")
                        
                        if not available_slots:
                            print("\nNo slots found, checking alternative dates...")
                            alt_dates = []
                            for i in range(1, 4):
                                alt_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
                                print(f"Checking alternative date: {alt_date}")
                                cursor.execute(query, [alt_date, position])
                                alt_rows = cursor.fetchall()
                                if alt_rows:
                                    alt_dates.append(alt_date)
                                    print(f"Found available slots for {alt_date}")
                            
                            print(f"Alternative dates found: {alt_dates}")
                            return json.dumps({
                                "date": date,
                                "time_frame": time_frame,
                                "position": position,
                                "available_slots": [],
                                "suggested_dates": alt_dates,
                                "message": "No available slots found for the requested date. Suggested alternative dates provided."
                            })
                        
                        result = {
                            "date": date,
                            "time_frame": time_frame,
                            "position": position,
                            "available_slots": available_slots,
                            "total_slots": len(available_slots)
                        }
                        print(f"\nFinal result: {json.dumps(result, indent=2)}")
                        return json.dumps(result)
                        
                except Exception as db_error:
                    print(f"✗ Database error: {str(db_error)}")
                    return json.dumps({
                        "error": f"Database error: {str(db_error)}"
                    })
                    
            except Exception as e:
                print(f"✗ Error in query_available_slots: {str(e)}")
                return json.dumps({
                    "error": f"Error querying available slots: {str(e)}"
                })
        
        # Create the agent with the tool
        self.agent = create_react_agent(
            self.llm,
            tools=[query_available_slots],
            prompt=self.system_prompt,
            name="sched_agent"
        )
        self.logger.info("✓ Agent created successfully")

    def _get_db_connection(self):
        try:
            self.logger.info("🔌 Attempting database connection...")
            self.logger.debug("Database configuration:")
            for key, value in self.db_config.items():
                if key != 'pwd':  # Don't log the password
                    self.logger.debug(f"{key}: {value}")
            
            conn_str = ';'.join([f"{k}={v}" for k, v in self.db_config.items()])
            self.logger.debug(f"Connection string (without password): {conn_str.replace(self.db_config['pwd'], '****')}")
            
            # Add timeout to the connection
            conn = pyodbc.connect(conn_str, timeout=30)  # 30 second timeout
            log_system_health("SQL Server", "ALIVE", {
                "server": self.db_config['server'],
                "database": self.db_config['database']
            })
            return conn
        except Exception as e:
            error_msg = f"Database connection error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            log_system_health("SQL Server", "ERROR", {
                "error": str(e),
                "troubleshooting": [
                    "Check if database server is running and accessible",
                    "Verify database credentials are correct", 
                    "Check network connectivity to database server",
                    "Verify firewall settings"
                ]
            })
            raise

    def chat(self, user_message: str) -> str:
        print(f"\nProcessing chat message: {user_message}")
        try:
            # Create a human message with the user's input
            messages = [HumanMessage(content=user_message)]
            print("\n=== Agent Debug Information ===")
            print("1. Created messages for agent")
            
            # Invoke the agent with the message
            print("\n2. Invoking agent...")
            response = self.agent.invoke({"messages": messages})
            print("\n3. Raw agent response:")
            print(f"Response type: {type(response)}")
            print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            
            if isinstance(response, dict):
                if 'messages' in response:
                    print("\n4. Messages in response:")
                    for i, msg in enumerate(response['messages']):
                        print(f"\nMessage {i}:")
                        print(f"Type: {type(msg)}")
                        print(f"Content: {msg.content if hasattr(msg, 'content') else msg}")
                        if hasattr(msg, 'additional_kwargs'):
                            print(f"Additional kwargs: {msg.additional_kwargs}")
                
                if 'intermediate_steps' in response:
                    print("\n5. Intermediate steps:")
                    for i, step in enumerate(response['intermediate_steps']):
                        print(f"\nStep {i}:")
                        print(f"Action: {step[0] if isinstance(step, tuple) else step}")
                        if isinstance(step, tuple) and len(step) > 1:
                            print(f"Result: {step[1]}")
            
            # Get the last message from the response
            if 'messages' in response and response['messages']:
                final_response = response['messages'][-1].content
                print("\n6. Final response:")
                print(final_response)
                return final_response
            else:
                print("\nError: No messages in response")
                return "Error: No response from agent"
                
        except Exception as e:
            print(f"\n✗ Error in chat: {str(e)}")
            return f"Error: {str(e)}"

class SupervisorAgent:
    def __init__(self, model_name: str = "gpt-4o", conversation_id: str = None):
        """Initialize the supervisor agent with all sub-agents"""
        self.logger = logging.getLogger('supervisor_agent')
        self.logger.info("🎼 Initializing SupervisorAgent...")
        
        # Use provided conversation_id or generate a new one
        self.conversation_id = conversation_id if conversation_id else str(uuid.uuid4())
        log_conversation_event(self.conversation_id, "started", {"supervisor_initialized": True})
        
        self.memory = MemorySaver()
        
        # Initialize the language model for the supervisor
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True
        )
        log_system_health("Supervisor LLM", "INITIALIZED", {"model": model_name})
        
        # Create date-aware system prompt
        self.system_supervisor_prompt = self._create_date_aware_prompt()
# - Use "exit_agent" to end the conversation. You should use this agent if the candidate is not interested, not suitable, 
# has found another job, or has scheduled an interview. When using this agent, provide a summary of why the conversation is ending.


# - Use "exit_agent" if you underdtand from the converstaion he is not suited or intrested in the position or has scehduled an interview.
# send a summary of the conversation and then exit the chat.

        # Initialize all sub-agents
        self.info_agent = info_agent()
        # self.exit_agent = exit_agent()
        self.exit_agent = SmartExitAgent()
        self.sched_agent = sched_agent()
        log_system_health("All Sub-agents", "INITIALIZED", {
            "agents": ["info_agent", "smart_exit_agent", "sched_agent"]
        })
        
        # Create the supervisor workflow
        workflow = create_supervisor(
            agents=[self.info_agent.agent, self.exit_agent.agent, self.sched_agent.agent],
            model=self.llm,
            prompt=self.system_supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history"
        )
        
        # Compile the workflow for execution with a name
        self.workflow = workflow.compile(name="supervisor_workflow", checkpointer=self.memory)
        log_system_health("Supervisor Workflow", "INITIALIZED", {"name": "supervisor_workflow"})
    
    def _create_date_aware_prompt(self) -> str:
        """Create supervisor prompt with current date information"""
        from datetime import datetime
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_year = datetime.now().year
        current_day = datetime.now().strftime('%A')
#         return """
#         You are the Supervisor Agent for a Python Developer job application chatbot. Your job is to orchestrate the conversation between the candidate and three specialized agents: info_agent, sched_agent, and exit_agent.
# Your goals:
# Provide a warm, professional candidate experience.
# Guide the candidate toward scheduling an interview.
# Ensure all job-related answers are based on info_agent's retrievals from the knowledge base.
# Use the sched_agent for all interview scheduling.
# Use the exit_agent to determine when to end the conversation.

# Conversation Flow:
# Greeting: Start with a warm greeting, introduce yourself, and mention the Python Developer position.
# Interest Check: Ask if the candidate is interested in the position.
# Job Questions: If the candidate asks about the job, extract their question and use info_agent.
# After receiving the info_agent's response (which contains relevant context from the knowledge base), summarize the information to provide a clear, concise answer to the candidate.
# Do not simply relay the info_agent's response verbatim.
# Scheduling: If the candidate wants to schedule, extract the date/time/position and use sched_agent. Present available slots or alternatives as provided.
# Closure: After an interview is scheduled, summarize the details and ask if the candidate needs anything else.
# Ending: After each user message, send the full conversation to exit_agent. If it returns "END", reply with a polite goodbye and end the conversation. Never end the conversation on your own.
# Ambiguity: If a user message is unclear or contains multiple intents, ask clarifying questions or handle each intent in order.
# Agent Usage Rules:
# info_agent: Use only for job-related questions. After receiving the info_agent's output, summarize it in your own words for the candidate.
# sched_agent: Use only for checking or booking interview slots.
# exit_agent: Use only to determine if the conversation should end. Never end the conversation without its explicit "END".
# Important:
# Never answer job or scheduling questions yourself without first consulting the appropriate agent.
# When answering job-related questions, do not cite sources or include references—always provide a clear, summarized answer.
# Never insert [END] unless instructed by exit_agent.
# If the conversation ends, start a new session with a fresh greeting if the user returns.
# CURRENT DATE INFORMATION:
# - Today's date: {current_date}
# - Day of week: {current_day}
# - Current year: {current_year}
# """
        return f"""
You are a helpful chatbot assistant for the Python developer position.

CURRENT DATE INFORMATION:
- Today's date: {current_date}
- Day of week: {current_day}
- Current year: {datetime.now().year}

CONVERSATION FLOW:
- Start conversations with a warm greeting, introduce yourself, and ask if they're interested in the Python developer position.
- You aim to set an interview with the candidate so direct them into scheduling an interview, use shced_agent first to suggest an interview in the follwing day.
- Suggest to them to schedule an interview or ask them if they have any questions about the job
- If the candidate asking questions about the job DON'T ANSWER YOUSELF, use info_agent to retrieve the relevant information to use to answer the question.
- If the candidate wants to schedule an interview, use sched_agent to receieve availble slots.
- IMPORTANT after the candidate agrees for an interview in an availble slot, summarize the details of the interview information and ask if they need anything else.
- MUST - Uppon receiving [END] from exit_agent, greet the candidate and end the conversation.
- Don't insert [END] to the conversation by yourself, Let the exit_agent decide.

You have 3 agents at your disposal: "info_agent", "sched_agent", and "exit_agent".

info_agent: whenever the candidate wants to inquire about the job you should use this agent. 
-you MUST approach the agent with the question regarding the job the candidate has asked
Extract the question from the candidate's message and then use the info_agent to answer the question. 
IMPORTANT: When the info_agent provides an answer, you MUST relay the answer to the candidate.

sched_agent: This agent only provides availble slots for the candidate to schedule an interview.
- ALWAYS provide the agent with at least the date and optionally the preferred time frame if given. 
The agent should return available slots and suggest other slots if no available slots are found.
Don't use this agent besdides query for availble slots

exit_agent: this agent is used to determine if the conversation should end. 
Send the full conversation history to the agent and let it decide if the conversation should end.
The agent will return only "END" and "NOT END" as the result, 
if the result is "END" then you should end the conversation by only printing [END] to the candidate,
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
        from datetime import datetime
        self.logger.info(f"🗓️ Refreshed date context - Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
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
    
    def process_message(self, message: str) -> str:
        """Process a user message through the supervisor workflow"""
        try:
            self.logger.info(f"💬 Processing user message: {message[:100]}...")
            log_conversation_event(self.conversation_id, "message", {
                "direction": "user_to_supervisor",
                "message_length": len(message)
            })
            
            # Create initial message
            messages = [HumanMessage(content=message)]
            
            config = {"configurable": {"thread_id": self.conversation_id}}
            # Invoke the workflow
            response = self.workflow.invoke({"messages": messages}, config=config)
            
            # IMPORTANT: Log agent selection and message routing
            if 'intermediate_steps' in response:
                self.logger.info("🎯 === AGENT SELECTION AND MESSAGE ROUTING ===")
                for step in response['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) > 1:
                        action, result = step
                        if hasattr(action, 'name'):
                            # Log orchestrator decision to use specific agent
                            log_agent_communication(
                                'agent_communication',
                                self.conversation_id,
                                action.name,
                                "to_agent",
                                str(action.input),
                                "SUCCESS"
                            )
                            
                            # Log agent response back to orchestrator
                            log_agent_communication(
                                'agent_communication',
                                self.conversation_id,
                                action.name,
                                "from_agent",
                                str(result),
                                "SUCCESS"
                            )
            
            # Extract the final response
            if 'messages' in response and response['messages']:
                final_response = response['messages'][-1].content
                
                # Log full response for debugging (you can adjust length limit as needed)
                if len(final_response) <= 300:
                    self.logger.info(f"📤 Final supervisor response: {final_response}")
                else:
                    self.logger.info(f"📤 Final supervisor response: {final_response[:300]}... [TRUNCATED - Full length: {len(final_response)} chars]")
                
                log_conversation_event(self.conversation_id, "message", {
                    "direction": "supervisor_to_user",
                    "response_length": len(final_response)
                })
                
                return final_response
            else:
                error_msg = "Error: No response from supervisor"
                self.logger.error(f"❌ {error_msg}")
                log_system_health("Supervisor Response", "ERROR", {"error": error_msg})
                return error_msg
                
        except Exception as e:
            error_msg = f"Error in supervisor processing: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            log_system_health("Supervisor Processing", "ERROR", {"error": str(e)})
            return f"Error: {str(e)}"

def predict_convo_ending(text, model, tokenizer, device='cpu'):
    """Predict if conversation should end - same function from training script"""
    model.eval()
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    return {
        'prediction': "END" if prediction == 1 else "NOT_END",
        'confidence': confidence,
        'probabilities': {
            'NOT_END': probs[0][0].item(),
            'END': probs[0][1].item()
        }
    }

def main():
    # Setup main logger
    main_logger = logging.getLogger('main')
    main_logger.info("🚀 === Python Developer Job Assistant Starting ===")
    
    # Initialize supervisor once with initial conversation ID
    initial_conversation_id = str(uuid.uuid4())
    main_logger.info(f"📋 Initial Session ID: {initial_conversation_id}")
    
    try:
        supervisor = SupervisorAgent(conversation_id=initial_conversation_id)
        main_logger.info("✅ System initialization completed successfully")
        
        # Trigger initial conversation naturally
        initial_response = supervisor.process_message("Hello")
        print(f"Assistant: {initial_response}\n")
        
        while True:
            try:
                # Get user input
                user_message = input("Enter your message: ")
                main_logger.debug(f"📝 User input received: {user_message[:50]}...")
                
                # Process the message
                response = supervisor.process_message(user_message)
                
                # Check if conversation should end
                if response.strip() == "[END]":
                    main_logger.info("🔚 Conversation ending detected")
                    print("\n" + "="*50)
                    print("🔚 Conversation session ended")
                    print("🔄 Starting new session...")
                    print("="*50)
                    
                    # Start new conversation with fresh UUID (no agent reinitialization)
                    new_id = supervisor.start_new_conversation()
                    main_logger.info(f"🆕 New conversation session started: {new_id}")
                    
                    # Trigger natural greeting for new conversation
                    new_response = supervisor.process_message("Hello")
                    print(f"Assistant: {new_response}\n")
                    
            except KeyboardInterrupt:
                main_logger.info("👋 User initiated shutdown")
                log_conversation_event(supervisor.conversation_id, "ended", {"reason": "user_exit"})
                print("\n\n👋 Goodbye! Exiting the conversation system.")
                break
            except Exception as e:
                main_logger.error(f"❌ Error in conversation loop: {str(e)}")
                log_system_health("Main Loop", "ERROR", {"error": str(e)})
                print(f"\n❌ Error occurred: {str(e)}")
                print("🔄 Continuing with current session...")
                
    except Exception as e:
        main_logger.critical(f"💥 Critical error during system initialization: {str(e)}")
        log_system_health("System Initialization", "ERROR", {"error": str(e)})
        print(f"💥 Failed to initialize system: {str(e)}")
        return
    
    main_logger.info("🏁 Application shutdown complete")

if __name__ == "__main__":
    main()

