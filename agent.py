import time
import os
from typing import Any, Dict, Optional
from uagents import Agent, Context, Model
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Define your models
class TextRequest(Model):
    text: str
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024

class ImageAnalysisRequest(Model):
    image_url: str
    question: Optional[str] = "What's in this image?"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024

class LlamaResponse(Model):
    timestamp: int
    response: str
    model_used: str
    agent_address: str
    processing_time: float

class EndpointsResponse(Model):
    agent_name: str
    agent_address: str
    available_endpoints: Dict[str, str]

# Create your agent
agent = Agent(name="Groq Llama REST API", seed="groq_llama_agent_seed")

async def call_groq_text(text: str, temperature: float = 1.0, max_tokens: int = 1024) -> str:
    """Call Groq API for text completion"""
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

async def call_groq_image(image_url: str, question: str = "What's in this image?", 
                         temperature: float = 1.0, max_tokens: int = 1024) -> str:
    """Call Groq API for image analysis"""
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API for image analysis: {str(e)}"

# GET endpoint - Simple health check and agent info
@agent.on_rest_get("/health", LlamaResponse)
async def health_check(ctx: Context) -> Dict[str, Any]:
    """Health check endpoint"""
    ctx.logger.info("Health check requested")
    start_time = time.time()
    
    # Simple test with Groq
    test_response = await call_groq_text("Say 'I am healthy and ready to help!'")
    processing_time = time.time() - start_time
    
    return {
        "timestamp": int(time.time()),
        "response": test_response,
        "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
        "agent_address": ctx.agent.address,
        "processing_time": processing_time
    }

# POST endpoint for text completion
@agent.on_rest_post("/chat", TextRequest, LlamaResponse)
async def handle_text_completion(ctx: Context, req: TextRequest) -> LlamaResponse:
    """Handle text completion requests"""
    ctx.logger.info(f"Received text completion request: {req.text[:50]}...")
    start_time = time.time()
    
    # Call Groq API
    response = await call_groq_text(
        text=req.text,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    processing_time = time.time() - start_time
    ctx.logger.info(f"Response generated in {processing_time:.2f} seconds")
    
    return LlamaResponse(
        timestamp=int(time.time()),
        response=response,
        model_used="meta-llama/llama-4-scout-17b-16e-instruct",
        agent_address=ctx.agent.address,
        processing_time=processing_time
    )

# POST endpoint for image analysis
@agent.on_rest_post("/analyze-image", ImageAnalysisRequest, LlamaResponse)
async def handle_image_analysis(ctx: Context, req: ImageAnalysisRequest) -> LlamaResponse:
    """Handle image analysis requests"""
    ctx.logger.info(f"Received image analysis request for: {req.image_url}")
    start_time = time.time()
    
    # Call Groq API for image analysis
    response = await call_groq_image(
        image_url=req.image_url,
        question=req.question,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    processing_time = time.time() - start_time
    ctx.logger.info(f"Image analysis completed in {processing_time:.2f} seconds")
    
    return LlamaResponse(
        timestamp=int(time.time()),
        response=response,
        model_used="meta-llama/llama-4-scout-17b-16e-instruct",
        agent_address=ctx.agent.address,
        processing_time=processing_time
    )

# GET endpoint to list available endpoints
@agent.on_rest_get("/endpoints", EndpointsResponse)
async def list_endpoints(ctx: Context) -> EndpointsResponse:
    """List available endpoints"""
    ctx.logger.info("Endpoints list requested")
    
    return EndpointsResponse(
        agent_name=agent.name,
        agent_address=ctx.agent.address,
        available_endpoints={
            "GET /health": "Health check and test Groq connection",
            "GET /endpoints": "List all available endpoints",
            "POST /chat": "Text completion using Llama model",
            "POST /analyze-image": "Image analysis using Llama vision model"
        }
    )

if __name__ == "__main__":
    print(f"Starting {agent.name}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /endpoints - List endpoints")
    print("  POST /chat - Text completion")
    print("  POST /analyze-image - Image analysis")
    print("\nMake sure to set GROQ_API_KEY environment variable!")
    print("Agent will be available at: http://localhost:8000")
    
    agent.run()