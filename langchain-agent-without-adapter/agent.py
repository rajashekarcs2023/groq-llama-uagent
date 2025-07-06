"""
Simple Currency Agent with Chat Protocol Only
- LangGraph currency brain
- Chat protocol only (no structured handler)  
- mailbox=True for Agentverse registration
"""

import os
import httpx
from datetime import datetime
from uuid import uuid4

# Core uAgents imports
from uagents import Agent, Context, Protocol

# Chat protocol imports
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# LangChain imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ============================================================================
# CURRENCY TOOL
# ============================================================================

@tool
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """Use this to get current exchange rate."""
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()
        data = response.json()
        if 'rates' not in data:
            return {'error': 'Invalid API response format.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}

# ============================================================================
# GLOBAL LANGGRAPH AGENT
# ============================================================================

memory = MemorySaver()
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
tools = [get_exchange_rate]
currency_agent = create_react_agent(
    model,
    tools=tools,
    checkpointer=memory,
    prompt="You are a currency conversion assistant. Use the get_exchange_rate tool to provide accurate exchange rates."
)

# ============================================================================
# SIMPLE UAGENT WITH CHAT PROTOCOL ONLY
# ============================================================================

# Create the agent
agent = Agent(
    name="simple_currency_agent",
    port=8004,
    mailbox=True
)

print(f"Simple Currency Agent address: {agent.address}")

def create_text_chat(text: str) -> ChatMessage:
    """Create a chat message with text content"""
    content = [TextContent(type="text", text=text)]
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )

# Chat protocol setup
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages for currency conversion"""
    
    # Log the incoming message
    if msg.content and any(isinstance(item, TextContent) for item in msg.content):
        text_content = next((item for item in msg.content if isinstance(item, TextContent)), None)
        if text_content:
            ctx.logger.info(f"Got currency request from {sender}: {text_content.text}")
    
    # Send acknowledgment
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )

    # Process text content
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Processing currency query: {item.text}")
            
            try:
                # Use global LangGraph agent
                query = item.text
                config = {'configurable': {'thread_id': f"chat_{sender}"}}
                
                # Process the query using global currency agent
                result = await currency_agent.ainvoke({'messages': [('user', query)]}, config)
                
                # Extract and format the response
                response_text = "Unable to process your currency request."
                if result and 'messages' in result:
                    last_message = result['messages'][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        response_text = f"💱 {last_message.content}"
                
                # Send response back
                response_msg = create_text_chat(response_text)
                await ctx.send(sender, response_msg)
                ctx.logger.info(f"Sent currency response to {sender}")
                
            except Exception as e:
                ctx.logger.error(f"Error processing currency request: {e}")
                error_msg = create_text_chat(f"Sorry, I encountered an error: {str(e)}")
                await ctx.send(sender, error_msg)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgments"""
    ctx.logger.info(f"Received acknowledgment from {sender}")

# Include the chat protocol
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    print("🚀 Starting Simple Currency Agent...")
    print("💱 LangGraph currency brain activated")
    print("💬 Chat protocol only (no structured handler)")
    print("🔍 Auto-registered with Agentverse")
    agent.run()
