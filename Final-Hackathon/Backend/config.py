# config.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
# Load environment variables from .env file
load_dotenv()

# Fetch Gemini API key
# GEMINI_API_KEY ="AIzaSyCtXFm0H2Bq8DV8H6yS7Ov8C9ON7ufxQo8"
GEMINI_API_KEY ="AIzaSyBXAMtJ-2irD9xuwLYtelTLcvMi-ie0BKs"

# Validate presence of the key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

# Set Gemini model version
GEMINI_MODEL = "models/gemini-1.5-flash"


MONGO_URI = "mongodb://localhost:27017/Hackathon"
if not MongoClient:
    raise ValueError("Mongodb not connected")
client = MongoClient(MONGO_URI)


