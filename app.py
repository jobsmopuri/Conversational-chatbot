import streamlit as st
from typing import Annotated
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] =  os.getenv("GROQ_API_KEY")

# ---- State definition for LangGraph ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---- LangGraph definition ----
graph_builder = StateGraph(State)
llm = init_chat_model("groq:llama-3.1-8b-instant")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# ---- Streamlit App ----
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("💬 Conversational Chatbot...")

# Session state to store messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input for user prompt
user_input = st.chat_input("Type your message...")

# Display chat history
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# Process new user input
if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # Build LangGraph input state
    state = {"messages": [{"role": "user", "content": user_input}]}

    # Run the graph and get the bot response
    result = graph.invoke(state)
    response_msg = result["messages"][-1].content

    # Display assistant response
    st.chat_message("assistant").markdown(response_msg)
    st.session_state.chat_history.append(("assistant", response_msg))
