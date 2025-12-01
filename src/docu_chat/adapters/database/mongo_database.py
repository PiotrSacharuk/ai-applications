"""
MongoDB Database Implementation
"""
import pymongo
from typing import List, Tuple
from ...interfaces import DatabaseProvider
from ...config import MONGO_URL, MONGODB_DATABASE, MONGODB_COLLECTION


class MongoDatabase(DatabaseProvider):
    """MongoDB conversation storage implementation"""

    def __init__(self):
        self.client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
        self.db = self.client[MONGODB_DATABASE]
        self.collection = self.db[MONGODB_COLLECTION]

        # Create index for faster lookups
        try:
            self.collection.create_index([("session_id", 1)], unique=True)
        except:
            pass  # Index might already exist

    def load_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Load conversation history from MongoDB"""
        document = self.collection.find_one({"session_id": session_id})
        history = []

        if document:
            conversation = document.get("conversation", [])
            for i in range(0, len(conversation), 2):
                if i + 1 < len(conversation):
                    history.append((conversation[i], conversation[i + 1]))

        return history

    def save_history(self, session_id: str, messages: List[str]) -> None:
        """Save conversation messages to MongoDB"""
        document = self.collection.find_one({"session_id": session_id})

        if document:
            # Update existing conversation
            conversation = document.get("conversation", [])
            conversation.extend(messages)
            self.collection.update_one(
                {"session_id": session_id},
                {"$set": {"conversation": conversation}}
            )
        else:
            # Create new conversation
            self.collection.insert_one({
                "session_id": session_id,
                "conversation": messages
            })
