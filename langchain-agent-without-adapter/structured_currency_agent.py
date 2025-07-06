"""
Currency Agent with Structured Output using External Agent
Following business-calculator pattern:
- Chat protocol calls external AI agent for structured output
- StructuredOutputClientProtocol for external agent communication
- LangGraph currency brain for actual calculations
"""

import os
import httpx
from datetime import datetime
from uuid import uuid4
from typing import Any
from textwrap import dedent

# Core uAgents imports
from uagents import Agent, Context, Model, Protocol

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
# CURRENCY TOOL AND LANGGRAPH AGENT
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

# Global LangGraph agent
memory = MemorySaver()
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
tools = [get_exchange_rate]
currency_agent = create_react_agent(
    model,
    tools=tools,
    checkpointer=memory,
    prompt="You are a currency conversion assistant. Use the get_exchange_rate tool to provide accurate exchange rates."
)

async def perform_currency_calculation(currency_from: str, currency_to: str, amount: float = 1.0, sender: str = "unknown") -> str:
    """Perform currency conversion using LangGraph agent"""
    try:
        query = f"How much is {amount} {currency_from} worth in {currency_to}? Please calculate the exact converted amount."
        config = {'configurable': {'thread_id': f"chat_{sender}"}}
        
        print(f"🔍 DEBUG: Sending query to LangGraph: '{query}'")
        print(f"🔍 DEBUG: Using config: {config}")
        
        result = await currency_agent.ainvoke({'messages': [('user', query)]}, config)
        
        print(f"🔍 DEBUG: LangGraph result: {result}")
        
        if result and 'messages' in result:
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                return f"💱 CURRENCY CONVERSION 💱\\n\\n{last_message.content}"
        
        return f"💱 CURRENCY CONVERSION 💱\\n\\nUnable to convert {amount} {currency_from} to {currency_to}"
        
    except Exception as e:
        return f"💱 CURRENCY CONVERSION ERROR 💱\\n\\nError: {str(e)}"

# ============================================================================
# EXTERNAL AI AGENT CONFIGURATION (Following business-calculator pattern)
# ============================================================================

# Replace with an AI Agent Address that supports StructuredOutput
# OpenAI Agent: agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y
# Claude.ai Agent: agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx
AI_AGENT_ADDRESS = 'agent1qtlpfshtlcxekgrfcpmv7m9zpajuwu7d5jfyachvpa4u3dkt6k0uwwp2lct'

if not AI_AGENT_ADDRESS:
    raise ValueError("AI_AGENT_ADDRESS not set")

def create_text_chat(text: str) -> ChatMessage:
    """Create a chat message with text content"""
    content = [TextContent(type="text", text=text)]
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )

# ============================================================================
# CURRENCY CALCULATION MODELS
# ============================================================================

class CurrencyCalculationRequest(Model):
    currency_from: str
    currency_to: str
    amount: float = 1.0

# ============================================================================
# STRUCTURED OUTPUT PROTOCOL (Following business-calculator pattern)
# ============================================================================

class StructuredOutputPrompt(Model):
    prompt: str
    output_schema: dict[str, Any]

class StructuredOutputResponse(Model):
    output: dict[str, Any]

# ============================================================================
# UAGENT WITH STRUCTURED OUTPUT PATTERN
# ============================================================================

# Create the agent
agent = Agent(
    name="structured_currency_agent",
    port=8005,
    mailbox=True
)

print(f"Structured Currency Agent address: {agent.address}")

# Protocols
chat_proto = Protocol(spec=chat_protocol_spec)
struct_output_client_proto = Protocol(
    name="StructuredOutputClientProtocol", version="0.1.0"
)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages using external AI agent for structured output"""
    
    # Log incoming message
    if msg.content and any(isinstance(item, TextContent) for item in msg.content):
        text_content = next((item for item in msg.content if isinstance(item, TextContent)), None)
        if text_content:
            ctx.logger.info(f"Got currency request from {sender}: {text_content.text}")
    
    ctx.storage.set(str(ctx.session), sender)
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )

    # Process text content (matching simple agent pattern)
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Processing text message: {item.text}")
            ctx.storage.set(str(ctx.session), sender)
            
            # Create structured output prompt (following business-calculator pattern)
            prompt_text = dedent(f"""
                Extract the currency conversion parameters from this message:

                "{item.text}"
                
                The user wants to perform a currency conversion. Extract:
                1. The currency_from: Source currency code (e.g., "USD", "EUR", "GBP")
                2. The currency_to: Target currency code (e.g., "USD", "EUR", "GBP")
                3. The amount: Amount to convert (default: 1.0 if not specified)
                
                Examples:
                - "Convert 100 USD to EUR" -> currency_from: "USD", currency_to: "EUR", amount: 100.0
                - "What is GBP to JPY rate?" -> currency_from: "GBP", currency_to: "JPY", amount: 1.0
                - "Show me 50 euros in dollars" -> currency_from: "EUR", currency_to: "USD", amount: 50.0
                
                If you cannot determine the currencies, set currency_from: "USD" and currency_to: "EUR" as defaults.
                If no amount is specified, use 1.0 as the default amount.
            """)
            
            # Send structured output request to external AI agent (following business-calculator pattern)
            structured_prompt = StructuredOutputPrompt(
                prompt=prompt_text,
                output_schema=CurrencyCalculationRequest.schema()
            )
            
            ctx.logger.info(f"Sending structured output request to AI agent: {AI_AGENT_ADDRESS}")
            await ctx.send(AI_AGENT_ADDRESS, structured_prompt)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgments"""
    ctx.logger.info(f"Received acknowledgment from {sender}")

@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(
    ctx: Context, sender: str, msg: StructuredOutputResponse
):
    """Handle structured output response from external AI agent"""
    ctx.logger.info(f"Received structured output from {sender}: {msg.output}")
    
    try:
        # Extract parameters from structured output
        currency_from = msg.output.get("currency_from", "USD")
        currency_to = msg.output.get("currency_to", "EUR")
        amount = float(msg.output.get("amount", 1.0))
        
        ctx.logger.info(f"Extracted parameters: {amount} {currency_from} to {currency_to}")
        
        # Get the original sender from session storage
        session_sender = ctx.storage.get(str(ctx.session))
        
        # Perform currency calculation using LangGraph agent
        result = await perform_currency_calculation(currency_from, currency_to, amount, session_sender or "unknown")
        if session_sender:
            # Send result back to original sender
            response_message = create_text_chat(result)
            await ctx.send(session_sender, response_message)
            ctx.logger.info(f"Sent currency result to original sender: {session_sender}")
        else:
            ctx.logger.error("Could not find original sender in session storage")
            
    except Exception as e:
        ctx.logger.error(f"Error processing structured output: {e}")
        
        # Try to send error to original sender
        session_sender = ctx.storage.get(str(ctx.session))
        if session_sender:
            error_message = create_text_chat(f"Error processing currency request: {str(e)}")
            await ctx.send(session_sender, error_message)

# Include protocols
agent.include(chat_proto, publish_manifest=True)
agent.include(struct_output_client_proto, publish_manifest=True)

if __name__ == "__main__":
    print("🚀 Starting Structured Currency Agent...")
    print("💱 LangGraph currency brain activated")
    print("💬 Chat protocol with external AI agent for structured output")
    print(f"🤖 External AI Agent: {AI_AGENT_ADDRESS}")
    print("🔍 Auto-registered with Agentverse")
    agent.run()
