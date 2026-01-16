#!/usr/bin/env python3
"""
Test script for NocoAI API streaming endpoint.

This script tests the API at https://chatbot.nocoai.vn/api/stream to:
1. Verify the API is accessible and responds correctly
2. Understand the streaming response format
3. Extract conversation_id from the first response
4. Test subsequent calls with the conversation_id
"""

import asyncio
import json
import ssl
import certifi
import aiohttp


API_URL = "https://chatbot.nocoai.vn/api/stream"

# Fixed payload values as specified
PAYLOAD_TEMPLATE = {
    "model": "gpt-4o",
    "user": "YGQTM5JDxTW82zJwIf3kNrOgUfj2",
    "user_type": "premium",
    "source_type": "local",
    "storage": "document",
    "vector": "nhquan",
    "active_docs": "local/YGQTM5JDxTW82zJwIf3kNrOgUfj2/document/nhquan/",
    "project": "document",
}


async def call_api(question: str, conversation_id: str | None = None) -> tuple[str, str | None]:
    """
    Call the NocoAI API with a question.
    
    Args:
        question: The user's question
        conversation_id: Optional conversation ID from previous call
        
    Returns:
        Tuple of (full_response_text, conversation_id)
    """
    payload = {
        **PAYLOAD_TEMPLATE,
        "question": question,
        "conversation_id": conversation_id,
    }
    
    print(f"\n{'='*60}")
    print(f"Calling API with question: {question}")
    print(f"Conversation ID: {conversation_id}")
    print(f"{'='*60}")
    
    full_response = ""
    new_conversation_id = None
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(API_URL, json=payload) as response:
            print(f"Status: {response.status}")
            print(f"Content-Type: {response.content_type}")
            print(f"\nStreaming response chunks:")
            print("-" * 40)
            
            # Read the streaming response
            async for line in response.content:
                line_text = line.decode("utf-8").strip()
                if not line_text:
                    continue
                    
                print(f"Raw chunk: {line_text[:200]}{'...' if len(line_text) > 200 else ''}")
                
                # Try to parse as JSON to extract conversation_id
                try:
                    data = json.loads(line_text)
                    if isinstance(data, dict):
                        if "conversation_id" in data:
                            new_conversation_id = data["conversation_id"]
                            print(f"  -> Found conversation_id: {new_conversation_id}")
                        if "answer" in data:
                            full_response += data["answer"]
                        elif "content" in data:
                            full_response += data["content"]
                        elif "text" in data:
                            full_response += data["text"]
                        elif "delta" in data:
                            full_response += data.get("delta", "")
                except json.JSONDecodeError:
                    # If not JSON, might be SSE format
                    if line_text.startswith("data:"):
                        data_str = line_text[5:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                data = json.loads(data_str)
                                # Check for conversation_id in type:id format
                                if data.get("type") == "id" and "id" in data:
                                    new_conversation_id = data["id"]
                                    print(f"  -> Found conversation_id: {new_conversation_id}")
                                # Extract answer content
                                elif "answer" in data:
                                    full_response += data["answer"]
                                elif "content" in data:
                                    full_response += data["content"]
                            except json.JSONDecodeError:
                                full_response += data_str
                    else:
                        # Plain text response
                        full_response += line_text
            
            print("-" * 40)
    
    print(f"\nFull response: {full_response[:500]}{'...' if len(full_response) > 500 else ''}")
    print(f"Conversation ID returned: {new_conversation_id}")
    
    return full_response, new_conversation_id


async def main():
    print("=" * 60)
    print("NocoAI API Test Script")
    print("=" * 60)
    
    # Test 1: First call without conversation_id
    print("\n\n### TEST 1: Initial API call (no conversation_id)")
    response1, conv_id = await call_api("Xin chào, bạn là ai?")
    
    if not conv_id:
        print("\n⚠️  WARNING: No conversation_id returned from first call!")
        print("This might indicate a different response format than expected.")
    
    # Test 2: Follow-up call with conversation_id
    if conv_id:
        print("\n\n### TEST 2: Follow-up call (with conversation_id)")
        response2, conv_id2 = await call_api("Bạn có thể giúp gì cho tôi?", conv_id)
        
        print("\n\n### SUMMARY")
        print("=" * 60)
        print(f"First call conversation_id: {conv_id}")
        print(f"Second call conversation_id: {conv_id2}")
        print(f"Conversation IDs match: {conv_id == conv_id2}")
    else:
        print("\n\n### SUMMARY")
        print("=" * 60)
        print("Could not test conversation continuity - no conversation_id returned.")
        print("Please check the raw response format above to understand the API structure.")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
