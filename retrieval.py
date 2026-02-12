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

    if chat_history[video_id]:
        rewrite_messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history[video_id] + [
            HumanMessage(content=f"New question: {query}")
        ]
        model_rewrite = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=500
        )
        rewritten = model_rewrite.invoke(rewrite_messages).content.strip()
        search_query = rewritten
    else:
        search_query = query

    docs = vectordb.similarity_search(search_query, k=k)
    if not docs:
        return "No relevant documents found to answer your query."

    combined_context = "\n".join([f"- {doc.page_content}" for doc in docs])
    prompt = f"""Based on the following documents(they are captions of a video) and chat history, please answer this question: {query}

Documents:
{combined_context}

Please provide a clear, helpful answer using only the information from these documents and chat history.
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    model_answer = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1000
    )

    messages = [
        SystemMessage(content="You are a helpful assistant that remembers previous questions and answers."),
    ] + chat_history[video_id] + [
        HumanMessage(content=prompt)
    ]

    answer = model_answer.invoke(messages).content
    chat_history[video_id].append(HumanMessage(content=query))
    chat_history[video_id].append(AIMessage(content=answer))
    return answer, docs
