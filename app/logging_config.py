import logging
import logging.handlers
import os
from datetime import datetime
import json

class CustomFormatter(logging.Formatter):
    """Custom formatter that adds colors and better structure"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname for console
        if hasattr(record, 'color') and record.color:
            levelname_color = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        else:
            levelname_color = record.levelname
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Create formatted message
        formatted = f"[{timestamp}] [{levelname_color}] [{record.name}] {record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'conversation_id'):
            log_entry['conversation_id'] = record.conversation_id
        if hasattr(record, 'agent_name'):
            log_entry['agent_name'] = record.agent_name
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'status'):
            log_entry['status'] = record.status
        if hasattr(record, 'details'):
            log_entry['details'] = record.details
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """Setup comprehensive logging for the application"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console Handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = CustomFormatter()
    console_handler.setFormatter(console_formatter)
    
    # Add color flag to console records
    class ColorFilter(logging.Filter):
        def filter(self, record):
            record.color = True
            return True
    console_handler.addFilter(ColorFilter())
    
    # File Handler for general logs (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = CustomFormatter()
    file_handler.setFormatter(file_formatter)
    
    # JSON File Handler for structured logs (rotating)
    json_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'app_structured.json'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    json_handler.setLevel(logging.DEBUG)
    json_formatter = JSONFormatter()
    json_handler.setFormatter(json_formatter)
    
    # Agent Communication Handler (separate file for important agent interactions)
    agent_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'agent_communications.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    agent_handler.setLevel(logging.INFO)
    agent_formatter = CustomFormatter()
    agent_handler.setFormatter(agent_formatter)
    
    # Add filter for agent communications
    class AgentFilter(logging.Filter):
        def filter(self, record):
            return hasattr(record, 'agent_name') or 'agent' in record.name.lower()
    agent_handler.addFilter(AgentFilter())
    
    # Error Handler (separate file for errors only)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'errors.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = CustomFormatter()
    error_handler.setFormatter(error_formatter)
    
    # Add all handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(json_handler)
    root_logger.addHandler(agent_handler)
    root_logger.addHandler(error_handler)
    
    # Create specialized loggers
    create_specialized_loggers()
    
    logging.info("🚀 Logging system initialized successfully")
    logging.info(f"📁 Log files location: {os.path.abspath(log_dir)}")

def create_specialized_loggers():
    """Create specialized loggers for different components"""
    
    # Agent Communication Logger
    agent_logger = logging.getLogger('agent_communication')
    agent_logger.setLevel(logging.INFO)
    
    # System Health Logger
    health_logger = logging.getLogger('system_health')
    health_logger.setLevel(logging.INFO)
    
    # Database Logger
    db_logger = logging.getLogger('database')
    db_logger.setLevel(logging.INFO)
    
    # Vector Store Logger
    vector_logger = logging.getLogger('vector_store')
    vector_logger.setLevel(logging.INFO)
    
    # Conversation Logger
    conversation_logger = logging.getLogger('conversation')
    conversation_logger.setLevel(logging.INFO)

def log_agent_communication(logger_name, conversation_id, agent_name, direction, message, status="SUCCESS"):
    """Helper function to log agent communications with structured format"""
    logger = logging.getLogger(logger_name)
    
    extra = {
        'conversation_id': conversation_id,
        'agent_name': agent_name,
        'operation': f'agent_communication_{direction}',
        'status': status,
        'details': {
            'direction': direction,
            'message_length': len(message),
            'message_preview': message[:100] + "..." if len(message) > 100 else message
        }
    }
    
    if direction == "to_agent":
        logger.info(f"🎯 ORCHESTRATOR → {agent_name}: {message[:100]}{'...' if len(message) > 100 else ''}", extra=extra)
    elif direction == "from_agent":
        logger.info(f"📨 {agent_name} → ORCHESTRATOR: {message[:100]}{'...' if len(message) > 100 else ''}", extra=extra)

def log_system_health(component, status, details=None):
    """Helper function to log system health status"""
    logger = logging.getLogger('system_health')
    
    extra = {
        'operation': 'health_check',
        'status': status,
        'details': details or {}
    }
    
    if status == "ALIVE":
        logger.info(f"✅ {component} is healthy", extra=extra)
    elif status == "INITIALIZED":
        logger.info(f"🔧 {component} initialized successfully", extra=extra)
    elif status == "ERROR":
        logger.error(f"❌ {component} health check failed", extra=extra)
    elif status == "WARNING":
        logger.warning(f"⚠️  {component} has issues", extra=extra)

def log_conversation_event(conversation_id, event_type, details=None):
    """Helper function to log conversation events"""
    logger = logging.getLogger('conversation')
    
    extra = {
        'conversation_id': conversation_id,
        'operation': f'conversation_{event_type}',
        'status': 'INFO',
        'details': details or {}
    }
    
    if event_type == "started":
        logger.info(f"🆕 New conversation started: {conversation_id}", extra=extra)
    elif event_type == "ended":
        logger.info(f"🔚 Conversation ended: {conversation_id}", extra=extra)
    elif event_type == "message":
        logger.info(f"💬 Message in conversation: {conversation_id}", extra=extra) 