class LongTermMemory:

    def __init__(self):
        self.users = {}

    def get_memory(self, user_id: str):
        return self.users.get(
            user_id,
            {
                "preferences": {},
                "facts": [],
            }
        )

    def update_memory(
        self,
        user_id: str,
        preferences=None,
        facts=None,
    ):

        if user_id not in self.users:
            self.users[user_id] = {
                "preferences": {},
                "facts": [],
            }

        if preferences:
            self.users[user_id]["preferences"].update(
                preferences
            )

        if facts:
            self.users[user_id]["facts"].extend(facts)

        return self.users[user_id]


long_term_memory = LongTermMemory()