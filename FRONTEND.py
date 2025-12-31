import streamlit as st
from BACKEND import Chatbot, retrive_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# *********************** UTILITY FUNCTIONS ********************************************

def get_config(thread_id):
    """Generates the config dictionary dynamically to avoid KeyError."""
    return {'configurable': {'thread_id': str(thread_id)}}

def generate_thread_id():
    return str(uuid.uuid4())

def add_threads(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_threads(new_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    """Fetches messages from the backend for a specific thread."""
    state = Chatbot.get_state(config=get_config(thread_id))
    # If the thread is new, 'messages' key might not exist in values yet
    return state.values.get('messages', [])

# *********************** SESSION SETUP ********************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    # Assuming this returns a list from your backend
    st.session_state['chat_threads'] = retrive_all_threads() or []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    add_threads(st.session_state['thread_id'])

# *********************** SIDEBAR UI ********************************************
st.sidebar.title('LEJBOT')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    # --- CHANGE: Fetch first message content for the button label instead of ID ---
    history = load_conversation(thread_id)
    if history:
        # Use the first 25 characters of the first message as the label
        chat_label = f"{history[0].content[:25]}..."
    else:
        chat_label = "New Conversation"

    # Create a button for each thread using the chat content as the label
    if st.sidebar.button(chat_label, key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'assistant' if isinstance(msg, AIMessage) else 'user'
            temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages
        st.rerun() # Refresh UI to show the selected conversation

# *********************** MAIN UI ********************************************
# --- CHANGE: Removed Thread ID from the title ---
st.title("LEJBOT") 

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type Here')

if user_input:
    # 1. Display User Message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # 2. Generate and Stream AI Response
    with st.chat_message("assistant"):
        def ai_only_stream():
            # Use the dynamic config for the current thread
            current_config = get_config(st.session_state['thread_id'])
            
            for message_chunk, metadata in Chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=current_config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # 3. Save AI message to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})