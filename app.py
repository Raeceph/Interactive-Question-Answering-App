import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatDocument

st.set_page_config(page_title="ChatDocument")

def display_chat_messages() -> None:
    """Display chat messages from the session state."""
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state.get("messages", [])):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def handle_user_input() -> None:
    """Process user input and get a response from the assistant."""
    user_input = st.session_state.get("user_input", "").strip()
    if user_input:
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_response = st.session_state["assistant"].ask_question(user_input)
        st.session_state["messages"].append((user_input, True))
        st.session_state["messages"].append((agent_response, False))
        st.session_state["user_input"] = ""

def upload_and_process_files() -> None:
    """Upload and process the files."""
    st.session_state["assistant"].clear_data()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state.get("file_uploader", []):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest_file(file_path, file.name)
        os.remove(file_path)

def main_page() -> None:
    """Main page of the Streamlit app."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatDocument()

    st.header("ChatDocument")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf", "docx"],
        key="file_uploader",
        on_change=upload_and_process_files,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()
    display_chat_messages()
    st.text_input("Message", key="user_input", on_change=handle_user_input)

if __name__ == "__main__":
    main_page()
