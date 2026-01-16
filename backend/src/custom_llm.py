"""
Custom LLM implementation for NocoAI API.

This module provides a custom LLM class that integrates with LiveKit's voice pipeline
by calling the NocoAI streaming API at https://chatbot.nocoai.vn/api/stream
"""

from __future__ import annotations

import json
import logging
import ssl
from collections.abc import AsyncIterable
from typing import Any

import aiohttp
import certifi
from livekit.agents import llm

logger = logging.getLogger("nocoai-llm")


# API configuration
API_URL = "https://chatbot.nocoai.vn/api/stream"

# Fixed payload values as specified by the user
PAYLOAD_CONFIG = {
    "model": "gpt-4o",
    "user": "YGQTM5JDxTW82zJwIf3kNrOgUfj2",
    "user_type": "premium",
    "source_type": "local",
    "storage": "document",
    "vector": "nhquan",
    "active_docs": "local/YGQTM5JDxTW82zJwIf3kNrOgUfj2/document/nhquan/",
    "project": "document",
}


class NocoAILLM(llm.LLM):
    """
    Custom LLM that calls the NocoAI streaming API.
    
    This LLM maintains conversation continuity by tracking the conversation_id
    returned from the API and passing it back in subsequent requests.
    """

    def __init__(self) -> None:
        super().__init__()
        self._conversation_id: str | None = None
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        conn_options: Any = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Any = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> llm.LLMStream:
        """
        Start a chat completion with the NocoAI API.
        
        Args:
            chat_ctx: The chat context containing conversation history
            tools: Not supported by this LLM
            conn_options: Connection options (ignored)
            parallel_tool_calls: Not supported by this LLM
            tool_choice: Not supported by this LLM
            extra_kwargs: Additional kwargs (ignored)
            
        Returns:
            An LLMStream that yields chat chunks from the API
        """
        return NocoAILLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            conversation_id=self._conversation_id,
            ssl_context=self._ssl_context,
            conn_options=conn_options,
        )

    def set_conversation_id(self, conversation_id: str | None) -> None:
        """Update the conversation ID for subsequent requests."""
        self._conversation_id = conversation_id
        logger.debug(f"Conversation ID updated: {conversation_id}")


class NocoAILLMStream(llm.LLMStream):
    """
    Streaming response handler for the NocoAI API.
    
    Parses SSE events from the API and yields ChatChunk objects
    compatible with LiveKit's voice pipeline.
    """

    def __init__(
        self,
        *,
        llm: NocoAILLM,
        chat_ctx: llm.ChatContext,
        conversation_id: str | None,
        ssl_context: ssl.SSLContext,
        conn_options: Any,
    ) -> None:
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=None, conn_options=conn_options)
        self._nocoai_llm = llm
        self._conversation_id = conversation_id
        self._ssl_context = ssl_context
        self._collected_text = ""

    def _extract_user_question(self) -> str:
        """Extract the latest user message from the chat context."""
        # Get the last user message from chat context
        for msg in reversed(self._chat_ctx.items):
            if hasattr(msg, 'role') and msg.role == "user":
                if hasattr(msg, 'text_content'):
                    return msg.text_content or ""
                elif hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Handle list of content parts
                        for part in content:
                            if isinstance(part, str):
                                return part
                            elif hasattr(part, 'text'):
                                return part.text
        return ""

    async def _run(self) -> None:
        """Execute the API call and emit chat chunks."""
        question = self._extract_user_question()
        
        if not question:
            logger.warning("No user question found in chat context")
            return

        logger.info(f"Calling NocoAI API with question: {question[:100]}...")
        
        payload = {
            **PAYLOAD_CONFIG,
            "question": question,
            "conversation_id": self._conversation_id,
        }

        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(API_URL, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return

                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        if not line_text:
                            continue

                        # Parse SSE format: "data: {...}"
                        if line_text.startswith("data:"):
                            data_str = line_text[5:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue

                            try:
                                data = json.loads(data_str)
                                
                                # Extract conversation_id from type:id event
                                if data.get("type") == "id" and "id" in data:
                                    new_conv_id = data["id"]
                                    self._nocoai_llm.set_conversation_id(new_conv_id)
                                    logger.debug(f"Received conversation_id: {new_conv_id}")
                                
                                # Extract and emit answer chunks
                                elif "answer" in data:
                                    text_chunk = data["answer"]
                                    if text_chunk:
                                        self._collected_text += text_chunk
                                        # Emit a chat chunk with the text delta
                                        self._event_ch.send_nowait(
                                            llm.ChatChunk(
                                                id="nocoai",
                                                delta=llm.ChoiceDelta(
                                                    role="assistant",
                                                    content=text_chunk,
                                                ),
                                            )
                                        )
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE data: {e}")
                                continue

        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            raise

        logger.info(f"NocoAI response complete, length: {len(self._collected_text)}")
