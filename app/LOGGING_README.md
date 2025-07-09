# Comprehensive Logging System

This application now includes a comprehensive logging system that captures all relevant information about the application's operation.

## Log Files Created

The logging system creates several files in the `logs/` directory:

### 1. `app.log` - General Application Logs
- All application events and debug information
- Rotating file (10MB max, 5 backups)
- Format: `[timestamp] [level] [logger_name] message`

### 2. `app_structured.jsonl` - Structured JSON Logs  
- Same information as app.log but in JSON format
- One JSON object per line for easy parsing
- Includes metadata like conversation_id, agent_name, operation type
- Perfect for log analysis tools

### 3. `agent_communications.log` - **IMPORTANT Agent Interactions**
- **Orchestrator decisions to use specific agents**
- **Messages sent from orchestrator to agents** 
- **Agent responses back to orchestrator**
- Conversation flow between agents
- Rotating file (10MB max, 10 backups)

### 4. `errors.log` - Error-Only Logs
- All ERROR and CRITICAL level messages
- Failures and exceptions
- Rotating file (5MB max, 5 backups)

## What Gets Logged

### System Health & Initialization
- ✅ Agent initialization (info_agent, sched_agent, exit_agent, supervisor)
- ✅ Vector DB validation and document counts
- ✅ Documents uploaded to vector DB with chunk counts
- ✅ SQL server connection status 
- ✅ BERT model loading and warmup status
- ✅ LLM initialization for each agent

### Conversation Management  
- ✅ New chat sessions with unique UUIDs
- ✅ Conversation start/end events
- ✅ User message processing

### **Agent Communications (IMPORTANT)**
- ✅ **Orchestrator decision to use specific agents** (`🎯 ORCHESTRATOR → agent_name`)
- ✅ **Messages sent to agents** (with message preview)
- ✅ **Agent responses back to orchestrator** (`📨 agent_name → ORCHESTRATOR`)
- ✅ Agent selection reasoning

### Database Operations
- ✅ Database connection attempts and status
- ✅ SQL query execution (in sched_agent)
- ✅ Environment variable validation

### Document Processing
- ✅ PDF loading and processing
- ✅ Document chunking and embedding
- ✅ Vector store operations
- ✅ Document retrieval and search results

### Error Handling
- ✅ All exceptions and errors with stack traces
- ✅ System component failures
- ✅ Recovery attempts and fallbacks

## Key Logging Features

### Structured Logging
```python
log_agent_communication('agent_communication', conversation_id, 'info_agent', 'to_agent', message)
log_system_health('Vector Store', 'ALIVE', {'count': 150})
log_conversation_event(conversation_id, 'started', {'supervisor_initialized': True})
```

### Color-Coded Console Output
- 🚀 Initialization events
- ✅ Success events  
- ❌ Error events
- ⚠️ Warning events
- 🎯 Agent communications
- 💬 Message processing

### Performance Tracking
- Message lengths and processing times
- Agent response times
- Database query performance
- Document retrieval performance

## Usage Examples

### Finding Agent Communications
```bash
grep "ORCHESTRATOR →" logs/agent_communications.log
grep "→ ORCHESTRATOR" logs/agent_communications.log
```

### Finding Errors
```bash
tail -f logs/errors.log
```

### Analyzing Conversation Flow
```bash
grep "conversation_id_here" logs/app_structured.jsonl | jq .
```

### System Health Check
```bash
grep "health_check" logs/app_structured.jsonl | jq '.details'
```

## Log Levels Used

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about program execution  
- **WARNING**: Something unexpected happened but the program continues
- **ERROR**: A serious problem occurred but the program continues
- **CRITICAL**: A very serious error occurred that may stop the program

## Configuration

The logging system is configured in `logging_config.py` and initialized at the start of `main.py`. Log levels and file locations can be adjusted there.

## Troubleshooting

If logs aren't appearing:
1. Check that the `logs/` directory is created
2. Verify file permissions
3. Check the log level configuration
4. Look for initialization errors in the console output 