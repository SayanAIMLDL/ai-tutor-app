# import os
# import re
# import xml.etree.ElementTree as ET # <-- THE CRUCIAL IMPORT
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Dict, Any

# # --- NEW: Use pydantic_settings for robust configuration ---
# from .config import settings

# # --- LangChain Core Imports ---
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda

# # --- LangChain Community and Ollama Imports ---
# from langchain_community.vectorstores import Chroma # Will be updated to langchain_chroma
# from langchain_ollama import ChatOllama, OllamaEmbeddings


# # --- Load Models using the robust settings object ---
# llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)
# embedding_model = OllamaEmbeddings(model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)

# # --- Set up the Retriever ---
# project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# CHROMA_PATH = os.path.join(project_root, "chroma")
# vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
# retriever = vector_store.as_retriever()


# # --- Manual ReAct Prompt Template with XML format instructions ---
# react_prompt = PromptTemplate(
#     template="""
# <system>
# You are an expert AI research assistant. Your goal is to answer the user's question based on a specialized knowledge base.
# You must use the following tool to find information.

# <tools>
# <tool>
# <name>search_knowledge_base</name>
# <description>Searches the knowledge base for information about Artificial Intelligence, Machine Learning, RAG, and LangChain.</description>
# <parameters>
# <parameter>
# <name>query</name>
# <type>string</type>
# <description>The search query to find relevant information.</description>
# </parameter>
# </parameters>
# </tool>
# </tools>

# Follow this process:
# 1.  **Thought:** Analyze the user's question and decide if you need to use the tool. If the user is asking for a definition or explanation of an AI topic, you must use the tool.
# 2.  **Tool Call:** If you need to search, respond with a single, perfectly formatted `<tool_call>` XML block.
# 3.  **Synthesize:** Do not respond with a tool call if you already have the answer. If you have the search results, use them as context to provide a comprehensive, final answer.
# </system>

# <human>
# {question}
# </human>
# """,
#     input_variables=["question"],
# )

# # --- Functions for the Manual Agentic Loop ---
# def parse_tool_call(llm_output: str) -> Dict[str, Any]:
#     """
#     Parses a flexible <tool_call> XML block from the LLM's output using a robust XML parser.
#     """
#     try:
#         tool_call_match = re.search(r"<tool_call>.*?</tool_call>", llm_output, re.DOTALL)
#         if not tool_call_match:
#             print("DEBUG: No <tool_call> block found in LLM output.")
#             return {"final_answer": llm_output.strip()}
        
#         tool_call_xml = tool_call_match.group(0)
#         root = ET.fromstring(tool_call_xml)
        
#         name_element = root.find("name") or root.find("tool")
#         tool_name = name_element.text.strip() if name_element is not None else ""
        
#         query = ""
#         query_element = root.find("query") or root.find("value")
#         if query_element is not None:
#             query = query_element.text.strip()
#         else:
#             param_element = root.find("parameter")
#             if param_element is not None and "query" in param_element.attrib:
#                 query = param_element.attrib["query"]

#         if tool_name == "search_knowledge_base" and query:
#             print(f"DEBUG: Successfully parsed tool call for '{tool_name}' with query: '{query}'")
#             return {"tool_call": query}

#     except ET.ParseError as e:
#         print(f"DEBUG: XML parsing failed: {e}")
#         pass

#     print("DEBUG: Could not parse a valid tool call. Returning as final answer.")
#     return {"final_answer": llm_output.strip()}


# def execute_tool(agent_state: Dict[str, Any]) -> Dict[str, Any]:
#     print("---EXECUTING TOOL---")
#     query = agent_state.get("tool_call")
#     search_results = retriever.invoke(query)
#     return {"tool_results": str(search_results)}


# def generate_final_answer(agent_state: Dict[str, Any]) -> Dict[str, Any]:
#     print("---GENERATING FINAL ANSWER---")
#     context = agent_state.get("tool_results")
#     question = agent_state.get("question")
    
#     final_prompt_template = PromptTemplate(
#         template="""Based ONLY on the following context, provide a comprehensive, direct answer to the user's question. Be concise and helpful. Do not mention your thought process or that you used a tool. Context: {context} Question: {question}""",
#         input_variables=["context", "question"]
#     )
    
#     final_chain = final_prompt_template | llm
#     final_answer = final_chain.invoke({"context": context, "question": question})
#     return {"final_answer": final_answer.content}


# def run_agentic_chain(state: Dict[str, Any]) -> Dict[str, Any]:
#     print("---AGENT REASONING---")
#     llm_response = llm.invoke(react_prompt.format(question=state.get("question")))
#     parsed_response = parse_tool_call(llm_response.content)
    
#     if "tool_call" in parsed_response and parsed_response["tool_call"]:
#         print("---DECISION: Tool call detected.---")
#         tool_results = execute_tool(parsed_response)
#         final_state = {**state, **tool_results}
#         final_response = generate_final_answer(final_state)
#         return final_response
#     else:
#         print("---DECISION: No tool call detected, returning direct answer.---")
#         return parsed_response

# # --- FastAPI Web Server ---
# app = FastAPI(
#     title="AI Study Assistant API (Manual ReAct)",
#     version="1.1.0"
# )

# class Query(BaseModel):
#     text: str

# @app.post("/ask", summary="Ask a question to the ReAct agent")
# async def ask_question(query: Query):
#     response = run_agentic_chain({"question": query.text})
#     return {"answer": response.get("final_answer")}




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