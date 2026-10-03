from typing import Dict, List


class ConversationMemory:
    """
    Simple in-memory conversation history.

    Later we can replace this with Redis or MongoDB
    for production use.
    """

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.sessions: Dict[str, List[dict]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

        # Keep only the latest messages
        self.sessions[session_id] = (
            self.sessions[session_id][-self.max_messages:]
        )

    def get_history(
        self,
        session_id: str,
    ) -> List[dict]:

        return self.sessions.get(
            session_id,
            []
        )

    def clear(
        self,
        session_id: str,
    ):

        self.sessions.pop(
            session_id,
            None
        )