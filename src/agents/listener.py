from explain.codec import sem_from_code

class Listener:
    def act(self, msg_code: str, candidates: list[dict]) -> int:
        target_sem = sem_from_code(msg_code)
        for i, c in enumerate(candidates):
            if c == target_sem:
                return i
        return 0
