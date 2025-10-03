import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentExecutor
from langchain.schema import HumanMessage, AIMessage
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import uuid
from datetime import datetime
import json
import time
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable LLM caching for faster responses
set_llm_cache(InMemoryCache())

# Custom CSS for professional design
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding/margins */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 1.5rem 0;
        margin-bottom: 0;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
    }
    
    .header-subtitle {
        color: #6b7280;
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    .sidebar-title {
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    /* Session buttons */
    .session-btn {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        width: 100%;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s;
        color: #374151;
    }
    
    .session-btn:hover {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    
    .session-btn.active {
        background: #eff6ff;
        border-color: #3b82f6;
        color: #1d4ed8;
    }
    
    /* Chat container */
    .chat-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .message {
        margin-bottom: 1rem;
        display: flex;
    }
    
    .message.user {
        justify-content: flex-end;
    }
    
    .message-content {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 12px;
        line-height: 1.5;
    }
    
    .message.user .message-content {
        background: #3b82f6;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message.bot .message-content {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 4px;
    }
    
    .message-label {
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 4px;
        opacity: 0.7;
    }
    
    .message-tools {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 4px;
    }
    
    /* Input area */
    .input-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #2563eb;
    }
    
    /* Status indicators */
    .status {
        font-size: 0.875rem;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .status.connected {
        background: #dcfce7;
        color: #166534;
    }
    
    .status.error {
        background: #fee2e2;
        color: #dc2626;
    }
    
    /* Thinking indicator */
    .thinking {
        background: #f3f4f6;
        padding: 8px 12px;
        border-radius: 8px;
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Rate Limiter Class
class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.time_window = time_window
    
    def check_limit(self, session_id):
        now = time.time()
        # Clean old requests
        self.requests[session_id] = [
            t for t in self.requests[session_id] 
            if now - t < self.time_window
        ]
        
        if len(self.requests[session_id]) >= self.max_requests:
            return False, f"Rate limit exceeded. Please wait before sending more messages."
        
        self.requests[session_id].append(now)
        return True, ""

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.agent_executor = None
    st.session_state.chat_sessions = {}
    st.session_state.current_session_id = None
    st.session_state.connection_status = "Not Connected"
    st.session_state.sidebar_collapsed = False
    st.session_state.rate_limiter = RateLimiter(max_requests=20, time_window=60)
    st.session_state.supabase = None

# Keys configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def validate_input(user_input: str) -> tuple:
    """Validate user input"""
    if not user_input or len(user_input.strip()) < 3:
        return False, "Query too short. Please provide more details (at least 3 characters)."
    
    if len(user_input) > 2000:
        return False, "Query too long. Please keep it under 2000 characters."
    
    # Check for potential dangerous patterns
    dangerous_patterns = ['__import__', 'exec(', 'eval(', 'os.system', 'subprocess']
    if any(pattern in user_input.lower() for pattern in dangerous_patterns):
        return False, "Invalid input detected. Please rephrase your question."
    
    return True, ""

def save_session_to_db(session_id, session_data):
    """Save session to Supabase"""
    try:
        if st.session_state.supabase is None:
            return
        
        # Prepare messages for JSON serialization
        messages_json = []
        for msg in session_data['messages']:
            msg_copy = msg.copy()
            if 'timestamp' in msg_copy:
                msg_copy['timestamp'] = msg_copy['timestamp'].isoformat()
            messages_json.append(msg_copy)
        
        st.session_state.supabase.table('chat_sessions').upsert({
            'id': session_id,
            'name': session_data['name'],
            'created_at': session_data['created_at'].isoformat(),
            'messages': json.dumps(messages_json),
            'updated_at': datetime.now().isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"Could not save session to database: {str(e)}")

def load_sessions_from_db():
    """Load all sessions from database"""
    try:
        if st.session_state.supabase is None:
            return {}
        
        response = st.session_state.supabase.table('chat_sessions').select('*').order('created_at', desc=True).execute()
        
        sessions = {}
        for session in response.data:
            session_id = session['id']
            messages = json.loads(session['messages']) if session['messages'] else []
            
            # Convert timestamp strings back to datetime
            for msg in messages:
                if 'timestamp' in msg and isinstance(msg['timestamp'], str):
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
            
            sessions[session_id] = {
                'id': session_id,
                'name': session['name'],
                'created_at': datetime.fromisoformat(session['created_at']),
                'messages': messages,
                'session_memory': [],
                'history': []
            }
            
            # Rebuild session memory from messages
            for msg in messages:
                if msg['type'] == 'user':
                    sessions[session_id]['session_memory'].append(HumanMessage(content=msg['content']))
                else:
                    sessions[session_id]['session_memory'].append(AIMessage(content=msg['content']))
        
        return sessions
    except Exception as e:
        st.warning(f"Could not load sessions from database: {str(e)}")
        return {}

@st.cache_resource
def initialize_agent():
    """Initialize the LangChain agent with caching"""
    try:
        # Connect to Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Reconnect to existing vector store
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents"
        )
        
        # LLM setup with streaming
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            streaming=False
        )
        
        # Create base retriever with better search parameters
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                
            }
        )
        
        # Add contextual compression for better retrieval
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # QA Chain for better answers
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=base_retriever,
            return_source_documents=True
        )
        
        def qa_with_sources(query):
            """Question answering with source tracking"""
            try:
                result = qa_chain.invoke({"query": query})
                return result["result"]
            except Exception as e:
                return f"Error retrieving information: {str(e)}"
        
        # Document search tool
        Retriver_tool = Tool(
            name="document_search",
            func=qa_with_sources,
            description="Search and answer questions based on uploaded documents. Use this for ANY question about companies, acquisitions, financial data, or specific information that might be in the documents.",
        )
        
        # General QA tool
        def general_qa(query):
            """General question answering"""
            try:
                return llm.invoke(query).content
            except Exception as e:
                return f"Error: {str(e)}"
        
        qa_tool = Tool(
            name="general_question",
            func=general_qa,
            description="Answer general knowledge questions NOT related to the uploaded documents.",
        )
        
        # Summary tool
        def summarize_text(text):
            """Summarize text"""
            try:
                prompt = f"Summarize the following concisely:\n\n{text}"
                return llm.invoke(prompt).content
            except Exception as e:
                return f"Error: {str(e)}"
        
        summary_tool = Tool(
            name="summarize",
            func=summarize_text,
            description="Summarize text or information.",
        )
        
        # Explanation tool
        def explain_concept(concept):
            """Explain concepts"""
            try:
                prompt = f"Explain clearly:\n\n{concept}"
                return llm.invoke(prompt).content
            except Exception as e:
                return f"Error: {str(e)}"
        
        explanation_tool = Tool(
            name="explain",
            func=explain_concept,
            description="Explain concepts or ideas in detail.",
        )
        
        tools = [Retriver_tool, qa_tool, summary_tool, explanation_tool]
        tool_names = ", ".join([tool.name for tool in tools])
        
        # Custom ReAct prompt
        react_prompt = PromptTemplate.from_template(
            """Answer the following question as best you can. You have access to the following tools:

{tools}

Use this format STRICTLY:

Thought: Think about what needs to be done
Action: The exact tool name from [{tool_names}]
Action Input: The specific input for the tool
Observation: The result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: Provide a clear, complete answer

IMPORTANT:
1. For questions about documents, companies, or data, ALWAYS use "document_search" FIRST
2. Action Input should be the question/text only - no quotes or special formatting
3. Always provide a Final Answer

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""
        ).partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=tool_names
        )
        
        # Create agent
        custom_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
        
        return custom_agent, tools, supabase, "Connected Successfully"
        
    except Exception as e:
        return None, None, None, f"Connection Error: {str(e)}"
def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
    
    # Initialize session data
    st.session_state.chat_sessions[session_id] = {
        "id": session_id,
        "name": session_name,
        "created_at": datetime.now(),
        "messages": [],
        "session_memory": [],
        "history": []
    }
    
    st.session_state.current_session_id = session_id
    
    # Save to database
    save_session_to_db(session_id, st.session_state.chat_sessions[session_id])
    
    return session_id

def get_recent_context(session_data, max_messages=10):
    """Get only recent messages to avoid context overflow"""
    recent_messages = session_data["session_memory"][-max_messages*2:] if len(session_data["session_memory"]) > max_messages*2 else session_data["session_memory"]
    return recent_messages

def get_agent_executor_for_session(session_id):
    """Get agent executor with session-specific memory"""
    if not st.session_state.initialized:
        return None
    
    session_data = st.session_state.chat_sessions[session_id]
    
    # Get recent context to avoid overwhelming the model
    recent_memory = get_recent_context(session_data, max_messages=8)
    
    # Create summary buffer memory for this session
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY),
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        max_token_limit=1000
    )
    
    # Restore recent session memory
    memory.chat_memory.messages = recent_memory
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=st.session_state.agent,
        tools=st.session_state.tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it follows the correct format.",
        return_intermediate_steps=True,
        max_iterations=5,
        max_execution_time=45,
    )
    
    return agent_executor


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_agent_response(agent_executor, user_input):
    """Get response with retry logic"""
    return agent_executor.invoke({"input": user_input})

def get_response_with_fallback(agent_executor, user_input):
    """Try multiple strategies if initial response fails"""
    try:
        # Primary attempt
        return get_agent_response(agent_executor, user_input)
    except Exception as e1:
        st.warning(f"Primary attempt failed, trying simplified approach...")
        try:
            # Fallback 1: Try with simpler prompt
            simplified_input = f"Please answer briefly: {user_input}"
            return agent_executor.invoke({"input": simplified_input})
        except Exception as e2:
            st.warning(f"Simplified approach failed, using direct LLM...")
            try:
                # Fallback 2: Direct LLM call without tools
                llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
                response_content = llm.invoke(user_input).content
                return {"output": response_content, "intermediate_steps": []}
            except Exception as e3:
                raise Exception(f"All attempts failed: {str(e3)}")

def track_metrics(session_data):
    """Track conversation metrics"""
    total_messages = len(session_data["messages"])
    user_messages = sum(1 for m in session_data["messages"] if m["type"] == "user")
    bot_messages = total_messages - user_messages
    
    # Calculate session duration
    if session_data["messages"]:
        first_msg = session_data["messages"][0]["timestamp"]
        last_msg = session_data["messages"][-1]["timestamp"]
        duration = (last_msg - first_msg).seconds
    else:
        duration = 0
    
    return {
        "total_messages": total_messages,
        "user_messages": user_messages,
        "bot_messages": bot_messages,
        "session_duration": duration
    }

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Document Assistant</h1>
        <p class="header-subtitle">Intelligent document analysis powered by LangChain</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent if not done
    if not st.session_state.initialized:
        with st.spinner("Initializing AI Agent..."):
            agent, tools, supabase, status = initialize_agent()
            if agent is not None:
                st.session_state.agent = agent
                st.session_state.tools = tools
                st.session_state.supabase = supabase
                st.session_state.connection_status = status
                st.session_state.initialized = True
                
                # Load existing sessions from database
                loaded_sessions = load_sessions_from_db()
                if loaded_sessions:
                    st.session_state.chat_sessions = loaded_sessions
                    st.session_state.current_session_id = list(loaded_sessions.keys())[0]
            else:
                st.session_state.connection_status = status
    
    # Sidebar for session management
    with st.sidebar:
        st.markdown('<p class="sidebar-title">💬 Chat Sessions</p>', unsafe_allow_html=True)
        
        # Connection status
        status_class = "connected" if st.session_state.connection_status == "Connected Successfully" else "error"
        status_text = "🟢 Connected" if status_class == "connected" else f"🔴 {st.session_state.connection_status}"
        st.markdown(f'<div class="status {status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # New session button
        if st.button("+ New Chat", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # Display sessions
        if st.session_state.chat_sessions:
            for session_id, session_data in st.session_state.chat_sessions.items():
                is_active = session_id == st.session_state.current_session_id
                
                if st.button(
                    f"{session_data['name']}\n{len(session_data['messages'])} messages",
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()
        
        # Session actions
        if st.session_state.current_session_id:
            st.markdown("---")
            
            # Rename session
            new_name = st.text_input(
                "Session Name:", 
                value=st.session_state.chat_sessions[st.session_state.current_session_id]["name"]
            )
            if st.button("Save Name", key="save_name"):
                st.session_state.chat_sessions[st.session_state.current_session_id]["name"] = new_name
                save_session_to_db(st.session_state.current_session_id, st.session_state.chat_sessions[st.session_state.current_session_id])
                st.success("Name updated!")
                st.rerun()
            
            # Show session metrics
            if st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]:
                metrics = track_metrics(st.session_state.chat_sessions[st.session_state.current_session_id])
                st.markdown("---")
                st.markdown("**Session Stats:**")
                st.text(f"📊 Messages: {metrics['total_messages']}")
                st.text(f"⏱️ Duration: {metrics['session_duration']}s")
            
            # Delete session
            if len(st.session_state.chat_sessions) > 1:
                st.markdown("---")
                if st.button("🗑️ Delete Chat", key="delete_session"):
                    # Delete from database
                    if st.session_state.supabase:
                        try:
                            st.session_state.supabase.table('chat_sessions').delete().eq('id', st.session_state.current_session_id).execute()
                        except:
                            pass
                    
                    del st.session_state.chat_sessions[st.session_state.current_session_id]
                    st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()
    
    # Main content
    if not st.session_state.initialized:
        st.error("⚠️ Agent initialization failed. Please check your configuration.")
        return
    
    # Create default session if none exists
    if not st.session_state.chat_sessions:
        create_new_session()
    
    # Ensure current session exists
    if st.session_state.current_session_id not in st.session_state.chat_sessions:
        st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
    
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    
    # Chat messages display
    if current_session["messages"]:
        for message in current_session["messages"]:
            if message["type"] == "user":
                st.markdown(f'''
                <div class="message user">
                    <div class="message-content">
                        <div class="message-label">You</div>
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                tools_info = ""
                if message.get('tools_used'):
                    tools_info = f'<div class="message-tools">🔧 Tools: {", ".join(message["tools_used"])}</div>'
                
                sources_info = ""
                if message.get('sources'):
                    sources_info = f'<div class="message-tools">📚 Sources: {len(message["sources"])} documents</div>'
                
                st.markdown(f'''
                <div class="message bot">
                    <div class="message-content">
                        <div class="message-label">Assistant</div>
                        {message["content"]}
                        {tools_info}
                        {sources_info}
                        
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; color: #6b7280; padding: 2rem;">
            👋 Start a conversation by asking a question about your documents
        </div>
        """, unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_area(
                "Message",
                placeholder="Ask a question about your documents...",
                height=80,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Send", use_container_width=True)
    
    # Process user input
    if submit_button and user_input.strip():
        # Validate input
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Check rate limit
        can_proceed, rate_limit_msg = st.session_state.rate_limiter.check_limit(st.session_state.current_session_id)
        if not can_proceed:
            st.error(rate_limit_msg)
            return
        
        # Add user message to session
        user_message = {
            "type": "user",
            "content": user_input,
            "timestamp": datetime.now()
        }
        current_session["messages"].append(user_message)
        current_session["session_memory"].append(HumanMessage(content=user_input))
        
        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown('<div class="thinking">🤔 Thinking...</div>', unsafe_allow_html=True)
        
        try:
            # Get agent executor for current session
            agent_executor = get_agent_executor_for_session(st.session_state.current_session_id)
            
            # Get response from agent with fallback
            response = get_response_with_fallback(agent_executor, user_input)
            answer = response["output"]
            
            # Extract tools used
            tools_used = []
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if len(step) > 0 and hasattr(step[0], 'tool'):
                        tools_used.append(step[0].tool)
            tools_used = list(set(tools_used)) if tools_used else []
            
            # Add bot message to session
            bot_message = {
                "type": "bot",
                "content": answer,
                "timestamp": datetime.now(),
                "tools_used": tools_used
            }
            current_session["messages"].append(bot_message)
            current_session["session_memory"].append(AIMessage(content=answer))
            
            # Save session to database
            save_session_to_db(st.session_state.current_session_id, current_session)
            
        except Exception as e:
            error_message = {
                "type": "bot",
                "content": f"❌ I encountered an error processing your request. Please try rephrasing your question or try again later.\n\nError: {str(e)}",
                "timestamp": datetime.now()
            }
            current_session["messages"].append(error_message)
            current_session["session_memory"].append(AIMessage(content=error_message["content"]))
        
        finally:
            thinking_placeholder.empty()
            st.rerun()

if __name__ == "__main__":
    main()