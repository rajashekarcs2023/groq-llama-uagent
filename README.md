# groq-llama-uagent

Example Query: 

 curl -d '{"text": "Explain AI in simple terms"}' -H "Content-Type: application/json" -X POST http://localhost:8000/chat


 curl -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg", "question": "What do you see in this image?"}' -H "Content-Type: application/json" -X POST http://localhost:8000/analyze-image

 