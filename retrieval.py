from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

chat_history = {}

def answer_query(query, video_id, k=3):
    persist_dir = f"./chromadb/{video_id}"

    if video_id not in chat_history:
        chat_history[video_id] = []

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return "Embeddings not found. Please run the ingestion step first."

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    docs = vectordb.similarity_search(query, k=k)

    if not docs:
        return "No relevant documents found to answer your query."

    combined_context = "\n".join(
        [f"- {doc.page_content[:500]}" for doc in docs] 
    )

    prompt = f"""Based on the following video captions(they are captions of a youtube video) and chat history, please answer this question: {query}

Documents:
{combined_context}

Please provide a clear, helpful answer using only the information from these captions and chat history.
If you can't find the answer in the captions, say "I don't have enough information to answer that question based on the provided documents."
"""

    model = ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=250,
        temperature=0
    )

    limited_history = chat_history[video_id]

    messages = [
        SystemMessage(content="You answer questions based only on provided captions."),
    ] + limited_history + [
        HumanMessage(content=prompt)
    ]

    response = model.invoke(messages)
    answer = response.content

    chat_history[video_id].append(HumanMessage(content=query))
    chat_history[video_id].append(AIMessage(content=answer))

    return answer, docs
