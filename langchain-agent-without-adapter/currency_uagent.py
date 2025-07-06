"""
Direct uAgent wrapper for CurrencyAgent with Agentverse registration.
"""

import os
import asyncio
import threading
import time
import requests
from typing import Any, Dict
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Agent, Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatMessage, 
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec
)
from pydantic import BaseModel

# Import CurrencyAgent - adjust path as needed
try:
    from app.agent import CurrencyAgent
except ImportError:
    # If app.agent doesn't exist, you can put CurrencyAgent code directly here
    # or adjust the import path to match your file structure
    print("❌ Cannot import CurrencyAgent from app.agent")
    print("💡 Create app/agent.py with your CurrencyAgent code, or adjust the import path")
    exit(1)

# Message models
class QueryMessage(Model):
    query: str

class ResponseMessage(Model):
    response: str

# HTTP REST endpoint models (following uAgents documentation pattern)
class CurrencyRequest(Model):
    text: str  # Using 'text' field as per documentation pattern

class CurrencyResponse(Model):
    response: str
    agent_address: str
    timestamp: int

# Chat protocol setup
chat_proto = Protocol(spec=chat_protocol_spec)

class CurrencyUAgent:
    """uAgent wrapper for CurrencyAgent with Agentverse registration."""
    
    def __init__(self, name: str, port: int, api_token: str = None):
        self.name = name
        self.port = port
        self.api_token = api_token
        
        # Initialize the LangGraph currency agent
        self.currency_agent = CurrencyAgent()
        
        # Create uAgent with mailbox for Agentverse discovery
        self.uagent = Agent(
            name=name,
            port=port,
            seed=f"currency_{name}_{port}",
            mailbox=True  # Enable for Agentverse registration
        )
        
        self.agent_address = None
        self.is_running = False
        
        # Setup handlers
        self._setup_handlers()
        
        # Setup REST endpoints
        self._setup_rest_endpoints()
    
    def _setup_handlers(self):
        """Setup message handlers for the uAgent."""
        
        @self.uagent.on_event("startup")
        async def startup(ctx: Context):
            self.agent_address = ctx.agent.address
            self.is_running = True
            ctx.logger.info(f"Currency uAgent '{self.name}' started")
            ctx.logger.info(f"Address: {self.agent_address}")
        
        @self.uagent.on_message(model=QueryMessage)
        async def handle_query(ctx: Context, sender: str, msg: QueryMessage):
            """Handle direct query messages."""
            try:
                ctx.logger.info(f"Received query: {msg.query}")
                
                # Process through currency agent with streaming
                response_content = ""
                async for item in self.currency_agent.stream(msg.query, str(ctx.session)):
                    if item['is_task_complete']:
                        response_content = item['content']
                        break
                    elif item['require_user_input']:
                        response_content = item['content']
                        break
                
                if not response_content:
                    response_content = "Unable to process currency request."
                
                # Send response
                await ctx.send(sender, ResponseMessage(response=response_content))
                ctx.logger.info("Sent response")
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                ctx.logger.error(error_msg)
                await ctx.send(sender, ResponseMessage(response=error_msg))
        
        @chat_proto.on_message(ChatMessage)
        async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
            """Handle chat protocol messages for Agentverse discovery."""
            try:
                ctx.logger.info(f"Chat message from {sender}")
                
                # Send acknowledgment
                await ctx.send(sender, ChatAcknowledgement(
                    timestamp=datetime.now(timezone.utc),
                    acknowledged_msg_id=msg.msg_id
                ))
                
                # Process text content
                for item in msg.content:
                    if isinstance(item, TextContent):
                        ctx.logger.info(f"Processing text: {item.text}")
                        
                        # Process through currency agent
                        response_content = ""
                        async for stream_item in self.currency_agent.stream(item.text, str(ctx.session)):
                            if stream_item['is_task_complete']:
                                response_content = stream_item['content']
                                break
                            elif stream_item['require_user_input']:
                                response_content = stream_item['content']
                                break
                        
                        if not response_content:
                            response_content = "Unable to process your currency request."
                        
                        # Send chat response
                        chat_response = ChatMessage(
                            timestamp=datetime.now(timezone.utc),
                            msg_id=uuid4(),
                            content=[TextContent(type="text", text=response_content)]
                        )
                        await ctx.send(sender, chat_response)
                        
            except Exception as e:
                ctx.logger.error(f"Error in chat handler: {str(e)}")
                error_response = ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=f"Sorry, I encountered an error: {str(e)}")]
                )
                await ctx.send(sender, error_response)
        
        @chat_proto.on_message(ChatAcknowledgement)
        async def handle_chat_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            ctx.logger.info(f"Chat acknowledgment from {sender}")
        
        # Include chat protocol
        self.uagent.include(chat_proto)
        
    def _setup_rest_endpoints(self):
        """Setup REST endpoints after agent initialization."""
        # Add HTTP REST endpoint for A2A bridge communication  
        @self.uagent.on_rest_post("/currency", CurrencyRequest, CurrencyResponse)
        async def handle_http_request(ctx: Context, req: CurrencyRequest) -> CurrencyResponse:
            """Handle HTTP POST requests from A2A bridge."""
            try:
                ctx.logger.info(f"HTTP POST request received: {req.text}")
                
                # Process through currency agent
                response_content = ""
                async for item in self.currency_agent.stream(req.text, "http_session"):
                    if item['is_task_complete']:
                        response_content = item['content']
                        break
                    elif item['require_user_input']:
                        response_content = item['content']
                        break
                
                if not response_content:
                    response_content = "Unable to process currency request."
                
                ctx.logger.info(f"HTTP response: {response_content[:100]}...")
                return CurrencyResponse(
                    response=response_content,
                    agent_address=ctx.agent.address,
                    timestamp=int(time.time())
                )
                
            except Exception as e:
                ctx.logger.error(f"HTTP handler error: {str(e)}")
                return CurrencyResponse(
                    response=f"Error: {str(e)}",
                    agent_address=ctx.agent.address,
                    timestamp=int(time.time())
                )
    
    def start(self):
        """Start the uAgent in background thread."""
        def run_agent():
            self.uagent.run()
        
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        
        # Wait for startup
        max_wait = 30
        wait_count = 0
        while not self.is_running and wait_count < max_wait:
            time.sleep(0.5)
            wait_count += 1
        
        if self.is_running:
            print(f"✅ Currency uAgent '{self.name}' is running!")
            print(f"📍 Address: {self.agent_address}")
            
            # Register with Agentverse if API token provided
            if self.api_token:
                self._register_with_agentverse()
            
            return True
        else:
            print(f"❌ Failed to start uAgent '{self.name}'")
            return False
    
    def _register_with_agentverse(self):
        """Register the agent with Agentverse."""
        try:
            print(f"🔗 Registering '{self.name}' with Agentverse...")
            
            # Wait a bit for agent to be fully ready
            time.sleep(5)
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Connect to Agentverse
            connect_url = f"http://127.0.0.1:{self.port}/connect"
            connect_payload = {
                "agent_type": "mailbox", 
                "user_token": self.api_token
            }
            
            try:
                connect_response = requests.post(connect_url, json=connect_payload, headers=headers)
                if connect_response.status_code == 200:
                    print(f"✅ Connected '{self.name}' to Agentverse")
                else:
                    print(f"❌ Failed to connect to Agentverse: {connect_response.status_code}")
                    return
            except Exception as e:
                print(f"❌ Error connecting to Agentverse: {str(e)}")
                return
            
            # Update agent info on Agentverse
            update_url = f"https://agentverse.ai/v1/agents/{self.agent_address}"
            
            description = (
                "A specialized currency conversion agent that provides real-time exchange rates "
                "between different currencies using the Frankfurter API. Ask me about currency "
                "conversions, exchange rates, or currency information."
            )
            
            readme_content = f"""# {self.name}
![tag:currency](https://img.shields.io/badge/currency-blue)
![tag:exchange_rates](https://img.shields.io/badge/exchange_rates-green)

{description}

## Capabilities
- Real-time currency exchange rates
- Historical exchange rate data
- Currency conversion calculations
- Support for major world currencies

## Input Format
class QueryMessage(Model):
query: str

## Output Format
class ResponseMessage(Model):
response: str

## Output Format
class ResponseMessage(Model):
response: str
## Example Queries
- "What is the USD to EUR exchange rate?"
- "Convert 100 USD to JPY"
- "Show me EUR to GBP rate for 2024-01-01"
"""
            
            update_payload = {
                "name": self.name,
                "readme": readme_content,
                "short_description": description
            }
            
            try:
                update_response = requests.put(update_url, json=update_payload, headers=headers)
                if update_response.status_code == 200:
                    print(f"✅ Updated '{self.name}' info on Agentverse")
                    print(f"🌍 Agent discoverable at: https://agentverse.ai/agents/{self.agent_address}")
                else:
                    print(f"❌ Failed to update Agentverse info: {update_response.status_code}")
            except Exception as e:
                print(f"❌ Error updating Agentverse info: {str(e)}")
                
        except Exception as e:
            print(f"❌ Error in Agentverse registration: {str(e)}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "port": self.port,
            "address": self.agent_address,
            "is_running": self.is_running,
            "agentverse_registered": bool(self.api_token)
        }

# Usage
if __name__ == "__main__":
    # Get API token from environment
    API_TOKEN = os.getenv("AGENTVERSE_API_KEY")
    if not API_TOKEN:
        print("⚠️  Warning: No AGENTVERSE_API_KEY found. Agent will run without Agentverse registration.")
    
    # Create and start the currency uAgent
    currency_uagent = CurrencyUAgent(
        name="currency_exchange_agent",
        port=8081,
        api_token=API_TOKEN
    )
    
    if currency_uagent.start():
        print("\n🚀 Currency uAgent ready!")
        print("💱 Ready to handle currency conversion requests")
        
        if API_TOKEN:
            print("🔍 Discoverable on Agentverse network")
        
        print("\n📋 Example usage:")
        print("- Send QueryMessage with query: 'What is USD to EUR rate?'")
        print("- Use chat protocol for Agentverse discovery")
        print("- HTTP POST to http://localhost:8081/currency with {'text': 'What is USD to EUR rate?'}")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    else:
        print("❌ Failed to start currency uAgent")