"""
RAG chain: retrieval + Google Gemini LLM for generating responses.
"""

import os
from typing import AsyncGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

from .vector_store import similarity_search


SYSTEM_PROMPT = """You are AgriLink's friendly AI assistant. AgriLink is a social commerce platform connecting farmers directly with consumers in India.

Your role:
- Help users understand how to use the AgriLink platform
- Answer questions about buying, selling, orders, subscriptions, payments, and delivery
- Be warm, helpful, and concise
- If you don't know something, say so honestly and suggest contacting support
- Always answer in the language the user is writing in (English, Hindi, or Tamil)
- Keep responses short and to the point (2-4 sentences usually)
- Use bullet points for lists
- Never make up information not present in the context

Use the following context from our knowledge base to answer the user's question:

{context}

If the context doesn't contain relevant information for the question, say something like:
"I don't have specific information about that. Please contact our support team at support@agrilink.com or call 1800-123-4567 for help."
"""


def _build_chat_prompt():
    """Build the chat prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def _get_llm():
    """Get the Google Gemini LLM instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=512,
        convert_system_message_to_human=True,
    )


def _format_history(history: list[dict]) -> list:
    """Convert history dicts to LangChain message objects."""
    messages = []
    for msg in history[-10:]:  # Keep last 10 messages for context
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def _retrieve_context(question: str) -> str:
    """Retrieve relevant context from the vector store."""
    docs = similarity_search(question, k=4)
    if not docs:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content}")

    return "\n\n".join(context_parts)


def get_chat_response(question: str, history: list[dict] = None) -> str:
    """
    Get a non-streaming chat response.

    Args:
        question: The user's question.
        history: List of previous messages [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        The assistant's response as a string.
    """
    if history is None:
        history = []

    # Retrieve relevant context
    context = _retrieve_context(question)

    # Build the chain
    prompt = _build_chat_prompt()
    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        "context": context,
        "chat_history": _format_history(history),
        "question": question,
    })

    return response


async def get_streaming_response(question: str, history: list[dict] = None) -> AsyncGenerator[str, None]:
    """
    Get a streaming chat response (yields chunks of text).

    Args:
        question: The user's question.
        history: List of previous messages.

    Yields:
        Chunks of the response text.
    """
    if history is None:
        history = []

    # Retrieve relevant context
    context = _retrieve_context(question)

    # Build the chain
    prompt = _build_chat_prompt()
    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    # Stream response
    async for chunk in chain.astream({
        "context": context,
        "chat_history": _format_history(history),
        "question": question,
    }):
        yield chunk
