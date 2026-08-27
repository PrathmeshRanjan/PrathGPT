from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Database Configuration & Setup
# ---------------------------------------------------------------------------
# Ensure local data storage directory exists before creating the SQLite DB file
Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

# check_same_thread=False allows Streamlit's multi-threaded worker runtime
# to safely share the same SQLite database connection across different user actions
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Session factory for creating transactional database sessions
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Database Models (Tables)
# ---------------------------------------------------------------------------

class Conversation(Base):
    """
    Stores metadata for each conversation session/thread.
    Used by the Streamlit sidebar to list, name, and order past chats.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)  # Matches LangGraph thread_id
    title = Column(String, default="New Chat")           # Derived from the first user prompt
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)  # Used to sort chats by recency


class ChatMessage(Base):
    """
    Stores individual chat messages for UI rendering.
    Enables reloading the complete message history when switching threads.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)  # Associated conversation thread
    role = Column(String)                   # 'user' or 'assistant'
    content = Column(Text)                  # Text content of the message
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory(Base):
    """
    Stores persistent memories, user preferences, or facts across interactions.
    Used by memory tools to provide long-term context to the agent.
    """
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)  # Associated thread or user context
    memory = Column(Text)                   # The saved fact/note text
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database Operations / Helper Functions
# ---------------------------------------------------------------------------

def init_db():
    """Initializes and creates all database tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    """
    Creates a new conversation record or updates the timestamp of an existing one.
    If it's a new conversation, auto-generates a title from the first message (up to 40 chars).
    """
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            # Generate a readable title from the initial user prompt
            if first_message:
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.add(conversation)

        else:
            # Touch updated_at to bring active conversation to top of recent list
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def list_conversations():
    """Fetches all conversations sorted by most recently updated first (for the sidebar)."""
    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def save_chat_message(thread_id: str, role: str, content: str):
    """
    Persists a single chat message (user or assistant) and updates
    the parent conversation's last-updated timestamp.
    """
    db = SessionLocal()

    try:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )

        db.add(msg)

        # Update parent conversation's timestamp
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def get_chat_history(thread_id: str):
    """
    Retrieves all messages for a specific conversation in chronological order (oldest to newest).
    Used to repopulate the UI when a user selects a thread from the sidebar.
    """
    db = SessionLocal()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()


def save_memory(thread_id: str, memory: str):
    """Saves a long-term memory or user fact into the database for a given thread."""
    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=datetime.utcnow()
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()


def search_memory(thread_id: str, query: str):
    """
    Retrieves the most recent long-term memories (up to 20) for the thread
    and formats them as a clean bulleted list for the agent.
    """
    db = SessionLocal()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()