from retrieval import answer_query
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
def evaluate_retrieval(query, retrieved_chunks):
    eval_model = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=200
    )

    combined_text = "\n".join([f"- {chunk.page_content}" for chunk in retrieved_chunks])

    prompt = f"""
You are an evaluator. Given the user query and the retrieved document chunks, rate how relevant the retrieved chunks are to answering the query. 

User Query: {query}

Retrieved Chunks:
{combined_text}

Give a relevance score from 1 to 10 (10 = highly relevant, 1 = not relevant at all) and a brief reason. Return in this format:
Score: <1-10>
Reason: <brief justification>
"""

    messages = [
        SystemMessage(content="You are a strict evaluator of document relevance."),
        HumanMessage(content=prompt)
    ]

    result = eval_model.invoke(messages)
    return result.content
