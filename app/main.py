import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

# --- Import settings from our dedicated, robust config file ---
from .config import settings

# --- LangChain Core Imports ---
from langchain_core.prompts import PromptTemplate

# --- LangChain Imports for specific components ---
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


# --- Load Models using the settings object ---
# The powerful model for reasoning and final answer synthesis
llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

# The fast model for creating embeddings for retrieval
embedding_model = OllamaEmbeddings(model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)


# --- Set up the Retriever ---
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CHROMA_PATH = os.path.join(project_root, "chroma")
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
retriever = vector_store.as_retriever()


# --- ReAct Prompt Template ---
# This prompt encourages the LLM to perform the full ReAct process internally
react_prompt = PromptTemplate(
    template="""
<system>
You are an expert AI research assistant. Your goal is to answer the user's question based on a specialized knowledge base.
To do this, you must follow this process:

1.  **Thought:** Analyze the user's question and determine the best query to search the knowledge base.
2.  **Internal Search:** (You will simulate this) Use the conceptual query to search your knowledge base for relevant context.
3.  **Synthesize:** Based on the context you found, provide a comprehensive, direct, and final answer to the user's question. Be helpful and concise. Do not show your thought process or the tool calls in the final output. Simply provide the answer.
</system>

<human>
Based on the documents I have provided, please answer the following question: {question}
</human>
""",
    input_variables=["question"],
)


def extract_final_answer(llm_output: str) -> str:
    """
    Extracts the final answer from the LLM's full output, which might include
    its thought process. This function is designed to find the synthesized part.
    """
    # Markers that might indicate the start of the final answer
    synthesis_markers = [
        "Synthesize:", "Final Answer:", "Synthesis:", "Here is the answer:", "Ah, I have the results"
    ]

    # Find the latest occurrence of any marker
    start_index = -1
    for marker in synthesis_markers:
        found_index = llm_output.rfind(marker)
        if found_index > start_index:
            start_index = found_index + len(marker)

    # If a marker was found, take the text after it
    if start_index != -1:
        final_answer = llm_output[start_index:].strip()
        return final_answer
    
    # Fallback for models that might use XML tags
    tool_call_end = llm_output.rfind("</tool_call>")
    if tool_call_end != -1:
        # Take the text after the last tool_call block
        final_answer = llm_output[tool_call_end + len("</tool_call>"):].strip()
        # Clean up any leftover instructional phrases
        if final_answer.startswith("Please wait"):
            final_answer = "\n".join(final_answer.splitlines()[1:]).strip()
        return final_answer

    # If no specific markers are found, assume the entire output is the answer
    return llm_output.strip()


# --- FastAPI Web Server ---
app = FastAPI(
    title="AI Study Assistant API",
    version="1.2.0"
)

class Query(BaseModel):
    text: str

@app.post("/ask", summary="Ask a question to the AI agent")
async def ask_question(query: Query):
    """
    Receives a question, retrieves context, gets a synthesized answer from the LLM,
    and extracts the final, clean response.
    """
    print("---RETRIEVING CONTEXT---")
    # 1. Retrieve context first (RAG)
    context_docs = retriever.invoke(query.text)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    
    # 2. Create a new prompt with the retrieved context
    final_prompt = react_prompt.format(
        question=f"{query.text}\n\nUse the following context to form your answer:\n\n<context>{context_text}</context>"
    )

    print("---INVOKING LLM FOR FINAL ANSWER---")
    # 3. Get the full output from the LLM
    llm_response = llm.invoke(final_prompt)
    
    print("---EXTRACTING CLEAN ANSWER---")
    # 4. Use our function to clean up the response
    clean_answer = extract_final_answer(llm_response.content)
    
    return {"answer": clean_answer}
