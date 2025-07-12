# LLM User Simulator for Multi-Agent Job Interview System

## 🎯 **Project Overview**

Create an LLM agent to simulate diverse user conversations with the supervisor bot, covering all conversation scenarios and agent routing patterns for comprehensive system testing and validation.

---

## 📊 **System Analysis Completed**

### **Current System Architecture**
- **Supervisor Agent**: Orchestrates conversation flow between sub-agents
- **Info Agent**: RAG-based Q&A using Python Developer PDF (4 chunks, 1000 chars each)
- **Schedule Agent**: SQL database queries for interview slots (date, time_frame, position)
- **Exit Agent**: Fine-tuned OpenAI model + rule-based logic (returns END/NOT_END)

### **Available Job Information (PDF Content)**
✅ **Job Overview**: Passionate Python developer role at Tech company
✅ **Objectives**: Develop/test software, participate in SDLC, collaborate, write clean code
✅ **Tasks**: Data pipelines, code reviews, debugging, stay updated with trends
✅ **Required Skills**: 3+ years Python, Bachelor's CS/SE, frameworks (NumPy, Pandas, etc.)
✅ **Experience**: Front-end (HTML/CSS/JS), databases (SQL/NoSQL), problem-solving
✅ **Preferred Skills**: Django/Flask/Pyramid, ML/data science, cloud (AWS/GCP/Azure)
✅ **Company Benefits**: Industry-standard compensation, relocation assistance, growth opportunities

### **Logging System Ready**
✅ **Agent Communications**: Orchestrator decisions and message routing tracked
✅ **Conversation Flows**: UUIDs, start/end events, message processing
✅ **System Health**: Component status and error tracking
✅ **Structured Logs**: JSON format for automated parsing and validation

---

## 🎭 **User Personas & Characteristics**

### **1. Eager Candidate (High Interest)**
- **Temperature**: 0.3 (focused, consistent responses)
- **Traits**: Asks detailed questions, enthusiastic about scheduling
- **Language**: "I'm very interested!", "That sounds perfect!", "When can we schedule?"

### **2. Skeptical Candidate (Cautious)**
- **Temperature**: 0.7 (varied responses, asks follow-ups)
- **Traits**: Questions everything, needs convincing, multiple clarifications
- **Language**: "What exactly...", "Are you sure about...", "I need to know more..."

### **3. Direct Candidate (Time-Pressed)**
- **Temperature**: 0.2 (brief, to-the-point)
- **Traits**: Minimal questions, wants quick scheduling or quick exit
- **Language**: "Just tell me the salary", "Can we schedule today?", "Not interested"

### **4. Detail-Oriented Candidate (Thorough)**
- **Temperature**: 0.8 (creative, diverse questions)
- **Traits**: Asks about technologies, growth, company culture, benefits
- **Language**: "Can you elaborate on...", "What about...", "I'd like to understand..."

### **5. Indecisive Candidate (Uncertain)**
- **Temperature**: 0.9 (highly varied responses)
- **Traits**: Changes mind, asks same questions differently, hesitates
- **Language**: "I'm not sure...", "Maybe...", "Let me think about it..."

### **6. Disinterested Candidate (Not Looking)**
- **Temperature**: 0.4 (polite but firm)
- **Traits**: Polite rejection, already has job, not actively looking
- **Language**: "Thank you but...", "I'm not looking", "Already found something"

---

## 🔄 **Conversation Scenarios & Expected Agent Flows**

### **Scenario 1: Simple Greeting Flow**
```
User -> Supervisor -> User
Expected: Warm greeting, position introduction, interest check
```

### **Scenario 2: Question + Interest Flow**
```
User -> Supervisor -> Info Agent -> Supervisor -> User -> Supervisor -> Schedule Agent -> Supervisor -> User
Expected: Question answered, scheduling offered and completed
```

### **Scenario 3: Question + Not Interested Flow**
```
User -> Supervisor -> Info Agent -> Supervisor -> User -> Supervisor -> Exit Agent -> Supervisor
Expected: Question answered, polite disinterest, conversation ends
```

### **Scenario 4: Full Interview Flow**
```
User -> Supervisor -> Info Agent -> Supervisor -> User -> Supervisor -> Schedule Agent -> Supervisor -> User -> Supervisor -> Exit Agent -> Supervisor
Expected: Questions, scheduling, confirmation, natural ending
```

### **Scenario 5: Direct Scheduling Flow**
```
User -> Supervisor -> Schedule Agent -> Supervisor -> User -> Supervisor -> Exit Agent -> Supervisor
Expected: Immediate scheduling request, slots provided, booking confirmed
```

### **Scenario 6: Immediate Disinterest Flow**
```
User -> Supervisor -> User -> Supervisor -> Exit Agent -> Supervisor
Expected: Polite disinterest, quick conversation end
```

---

## 🛠 **Implementation Architecture**

### **Core Components**

#### **1. UserSimulator Class**
```python
class UserSimulator:
    def __init__(self, persona: str, temperature: float = 0.7):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        self.persona = persona
        self.conversation_history = []
        self.scenario_goals = []
        
    def generate_response(self, supervisor_message: str) -> str:
        # Generate contextual user response based on persona
        
    def should_continue(self) -> bool:
        # Determine if user should continue conversation
```

#### **2. ScenarioManager Class**
```python
class ScenarioManager:
    def __init__(self):
        self.scenarios = self._load_scenarios()
        self.question_bank = self._load_questions()
        
    def create_scenario(self, scenario_type: str, persona: str) -> ConversationScenario:
        # Create specific scenario with expected flow
        
    def generate_dates(self) -> List[str]:
        # Generate both specific and relative dates
```

#### **3. FlowValidator Class**
```python
class FlowValidator:
    def __init__(self):
        self.log_parser = LogParser()
        
    def extract_agent_flow(self, conversation_id: str) -> List[str]:
        # Parse logs to extract actual agent routing
        
    def validate_flow(self, expected: List[str], actual: List[str]) -> float:
        # Compare expected vs actual flow, return score
```

#### **4. TestRunner Class**
```python
class TestRunner:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.simulator = UserSimulator()
        self.validator = FlowValidator()
        
    def run_scenario(self, scenario: ConversationScenario) -> TestResult:
        # Execute full conversation scenario
        
    def generate_report(self, results: List[TestResult]) -> TestReport:
        # Create comprehensive test report
```

---

## 📋 **Question Bank Categories**

### **Technical Questions**
- "What programming languages will I be working with?"
- "Do you use any specific Python frameworks like Django or Flask?"
- "What databases do you work with?"
- "Is there machine learning work involved?"
- "What cloud platforms do you use?"

### **Role & Responsibilities**
- "What would be my main daily tasks?"
- "Will I be working on data pipelines?"
- "Is there code review process?"
- "How much debugging work is involved?"
- "Do you work with cross-functional teams?"

### **Experience & Qualifications**
- "How many years of experience do you need?"
- "Is a computer science degree required?"
- "What kind of portfolio should I have?"
- "Do you need front-end experience too?"

### **Company & Culture**
- "What's the work environment like?"
- "Do you offer remote work?"
- "What are the growth opportunities?"
- "What benefits do you provide?"
- "Is there relocation assistance?"

### **Compensation**
- "What's the salary range for this position?"
- "Do you offer equity or bonuses?"
- "What about vacation time?"

---

## 📅 **Date Handling Strategy**

### **Specific Dates**
```python
# Generate realistic dates
dates = [
    "2025-07-10",  # Tomorrow
    "2025-07-15",  # Next week
    "2025-07-22",  # Following week
    "2025-08-01",  # Next month
]
```

### **Relative Dates**
```python
# Natural language dates
relative_dates = [
    "tomorrow",
    "next Monday",
    "in 3 days",
    "next week",
    "the day after tomorrow",
    "this Friday",
    "in a week",
]
```

### **Time Preferences**
```python
time_frames = ["morning", "afternoon", "evening", None]
```

---

## 🔍 **Validation & Scoring System**

### **Flow Accuracy Score (0-100)**
- **Perfect Match**: 100 points
- **Correct agents, wrong order**: 80 points  
- **Missing expected agents**: -20 points per agent
- **Unexpected agents**: -10 points per agent

### **Conversation Quality Score (0-100)**
- **Natural flow**: 25 points
- **Realistic responses**: 25 points
- **Scenario completion**: 25 points
- **Appropriate ending**: 25 points

### **Agent Performance Score (0-100)**
- **Info agent accuracy**: 25 points
- **Schedule agent functionality**: 25 points
- **Exit agent timing**: 25 points
- **Supervisor orchestration**: 25 points

---

## 🗂 **Test Scenarios Matrix**

| Scenario | Persona | Expected Agents | Questions | Scheduling | Expected End |
|----------|---------|-----------------|-----------|-------------|-------------|
| Quick Exit | Disinterested | Supervisor → Exit | 0 | No | END |
| Info Only | Skeptical | Supervisor → Info → Exit | 2-3 | No | END |
| Full Flow | Eager | Supervisor → Info → Schedule → Exit | 1-2 | Yes | END |
| Direct Schedule | Direct | Supervisor → Schedule → Exit | 0 | Yes | END |
| Question Heavy | Detail-Oriented | Supervisor → Info (multiple) → Schedule → Exit | 4-5 | Yes | END |
| Indecisive | Uncertain | Supervisor → Info → Exit | 1-2 | Maybe | END |

---

## 📈 **Success Metrics**

### **Coverage Metrics**
- ✅ All agent combinations tested
- ✅ All question categories covered  
- ✅ All scheduling scenarios tested
- ✅ All persona types simulated

### **Accuracy Metrics**
- ✅ Agent flow accuracy > 90%
- ✅ Conversation realism > 85%
- ✅ Scenario completion > 95%
- ✅ Log parsing accuracy > 98%

### **Performance Metrics**  
- ✅ Test execution time < 30 seconds per scenario
- ✅ Log processing time < 5 seconds
- ✅ Report generation time < 10 seconds

---

## 🚀 **Implementation Phases**

### **Phase 1**: Foundation (2-3 hours)
1. ✅ PDF content analysis (COMPLETED)
2. ⏳ UserSimulator class with persona support
3. ⏳ Basic question bank and scenario definitions
4. ⏳ Date generation utilities

### **Phase 2**: Core Testing (3-4 hours)
1. ⏳ ScenarioManager with full scenario matrix
2. ⏳ LogParser for agent flow extraction
3. ⏳ FlowValidator with scoring algorithms
4. ⏳ Basic TestRunner implementation

### **Phase 3**: Advanced Features (2-3 hours)
1. ⏳ Comprehensive persona behaviors
2. ⏳ Advanced conversation patterns
3. ⏳ Detailed validation metrics
4. ⏳ Rich reporting system

### **Phase 4**: Production Ready (1-2 hours)
1. ⏳ Performance optimization
2. ⏳ Error handling and recovery
3. ⏳ Documentation and examples
4. ⏳ Integration testing

---

## 📁 **File Structure**
```
tests/
├── user_simulator/
│   ├── __init__.py
│   ├── simulator.py          # UserSimulator class
│   ├── personas.py           # Persona definitions
│   ├── scenarios.py          # ScenarioManager
│   ├── validators.py         # FlowValidator
│   ├── runners.py            # TestRunner
│   ├── question_bank.py      # Questions by category
│   ├── date_utils.py         # Date generation
│   └── reports.py            # Report generation
├── logs_parser/
│   ├── __init__.py
│   └── parser.py             # Log parsing utilities
└── examples/
    ├── run_all_scenarios.py   # Full test suite
    ├── single_scenario.py     # Individual test
    └── validate_logs.py       # Log validation only
```

---

This implementation plan provides a complete roadmap for creating a sophisticated user simulator that will thoroughly test your multi-agent system across all conversation scenarios and validate the agent routing patterns.

**Ready to proceed with implementation?** 🚀 