import streamlit as st
import os
from ingestion import chunk, embedding_chunks
from retrieval import answer_query
from get_transcript import get_transcript

st.set_page_config(page_title="YouTube Video QA Chat", page_icon="💬")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
    padding: 20px;
    background-color: #f7f7f8;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.user-msg {
    background-color: #0a84ff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    margin-bottom: 10px;
    max-width: 80%;
    float: right;
    clear: both;
}
.bot-msg {
    background-color: #e5e5ea;
    color: black;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    margin-bottom: 10px;
    max-width: 80%;
    float: left;
    clear: both;
}
.clearfix {
    clear: both;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 YouTube video QA Chat")
st.info("📌 Enter the YouTube video ID (the part after `v=` in the URL). Example: for `https://www.youtube.com/watch?v=9ze8OyL5N5Y`, the ID is `9ze8OyL5N5Y`.")

video_id = st.text_input("Enter YouTube Video ID:", key="video_id_input")
if video_id:

    if video_id not in st.session_state.chat_history:
        st.session_state.chat_history[video_id] = []

    persist_dir = os.path.join("chromadb", video_id)
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        transcript = get_transcript(video_id)
        documents = chunk(transcript, video_id)
        embedding_chunks(documents, video_id=video_id)

    user_question = st.text_input("Ask a question about the video:", key=f"user_question_input_{video_id}")
    if user_question:
        st.session_state.chat_history[video_id].append({"role": "user", "content": user_question})
        answer, _ = answer_query(user_question, video_id=video_id)
        st.session_state.chat_history[video_id].append({"role": "bot", "content": answer})

with st.container():
    if video_id in st.session_state.chat_history:
        for chat in st.session_state.chat_history[video_id]:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-msg">{chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{chat["content"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
