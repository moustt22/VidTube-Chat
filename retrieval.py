from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

store = {}

def get_video_history(video_id: str) -> InMemoryChatMessageHistory:
    if video_id not in store:
        store[video_id] = InMemoryChatMessageHistory()
    return store[video_id]

def answer_query(query, video_id, k=3):
    persist_dir = f"./chromadb/{video_id}"

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return "Embeddings not found. Please run the ingestion step first.", []

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    docs = vectordb.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found to answer your query.", []

    combined_context = "\n".join([f"- {doc.page_content[:500]}" for doc in docs])

    system_message = SystemMessage(content="You answer questions using only the provided YouTube captions.")
    human_message = HumanMessage(content=f"""Question:
{query}

Documents:
{combined_context}

Answer clearly using only the captions and chat history.
If you can't find the answer, say "I don't have enough information to answer that question."
""")

    history = get_video_history(video_id)
    messages = [system_message] + history.messages + [human_message]

    llm = ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=250,
        temperature=0
    )

    response = llm.invoke(messages)
    history.add_message(human_message)
    history.add_message(AIMessage(content=response.content))

    return response.content, docs
