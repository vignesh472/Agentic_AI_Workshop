import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up model
generation_config = {
    "temperature": 0.3,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def generate_answer(query, context_chunks):
    """Generate answer using Gemini API with context chunks"""
    # Combine context chunks
    contexts = [chunk[0] for chunk in context_chunks]
    full_context = "\n\n".join(contexts)
    
    # Build prompt with instructions
    prompt = f"""
    You are an expert AI research assistant. Answer the user's question based ONLY on the provided context.
    If the answer is not in the context, say "I don't know" - do not make up answers.
    
    CONTEXT:
    {full_context}
    
    QUESTION: {query}
    
    ANSWER:
    """
    
    # Generate response using Gemini
    response = model.generate_content(prompt)
    
    # Extract sources
    sources = {}
    for text, filename, page_num in context_chunks:
        key = f"{filename} (Page {page_num})"
        sources[key] = sources.get(key, 0) + 1

    # Dummy token usage (Gemini doesn't provide token stats like OpenAI)
    token_usage = {
        "prompt_tokens": None,
        "completion_tokens": None
    }

    return response.text, list(sources.keys()), token_usage
