from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = "mongodb://localhost:27017/Hackathon"
client = MongoClient(MONGO_URI)
db = client.agentic_ai