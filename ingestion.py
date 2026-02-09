from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def chunk(transcript, video_id, max_chunk_seconds=30):
    documents = []
    current_chunk = []
    current_start = None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for snippet in transcript:
        if current_start is None:
            current_start = snippet.start

        current_chunk.append(snippet)
        current_end = snippet.start + snippet.duration

        if current_end - current_start >= max_chunk_seconds:
            chunk_text = " ".join([s.text for s in current_chunk])
            sub_chunks = text_splitter.split_text(chunk_text)
            for i, sub_chunk in enumerate(sub_chunks):
                documents.append(
                    Document(
                        page_content=sub_chunk,
                        metadata={
                            "video_id": video_id,
                            "chunk_id": f"{int(current_start * 100)}_{i}",
                            "start_time": current_start,
                            "end_time": current_end
                        }
                    )
                )
            current_chunk = []
            current_start = None

    if current_chunk:
        chunk_text = " ".join([s.text for s in current_chunk])
        sub_chunks = text_splitter.split_text(chunk_text)
        for i, sub_chunk in enumerate(sub_chunks):
            documents.append(
                Document(
                    page_content=sub_chunk,
                    metadata={
                        "video_id": video_id,
                        "chunk_id": f"{int(current_start * 100)}_{i}",
                        "start_time": current_start,
                        "end_time": current_chunk[-1].start + current_chunk[-1].duration
                    }
                )
            )

    return documents

def embedding_chunks(documents=None, video_id=None):
    if video_id is None:
        raise ValueError("video_id must be provided to create a unique vector store folder.")

    persist_dir = f"./chromadb/{video_id}"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    if os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        if documents is None:
            raise ValueError("No documents provided to create embeddings.")
        vectorstore = Chroma.from_documents(
            documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return vectorstore
