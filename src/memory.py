class ConversationMemory:
    def __init__(self):
        self.history = []

    def add(self, user, assistant):
        self.history.append({
            "user": user,
            "assistant": assistant
        })

    def get_recent(self, n=3):
        return self.history[-n:] if len(self.history) > 0 else []

    def get_context(self, k=5):
        """
        Returns formatted conversation context for LLM prompt
        """
        if not self.history:
            return ""

        recent = self.history[-k:]

        context = ""
        for turn in recent:
            context += f"User: {turn['user']}\n"
            context += f"Assistant: {turn['assistant']}\n"

        return context.strip()

    def clear(self):
        """
        Optional: clears memory (useful for resetting sessions)
        """
        self.history = []