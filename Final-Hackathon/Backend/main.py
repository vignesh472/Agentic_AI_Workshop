# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.evaluate import router as evaluation_router

app = FastAPI(
    title="Agentic Mastery Evaluator",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",  # Your frontend origin
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use ["*"] to allow all (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

# Include your evaluation routes
app.include_router(evaluation_router, prefix="/api/evaluate")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
