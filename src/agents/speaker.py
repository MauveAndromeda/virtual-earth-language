from explain.codec import code_from_sem, explain_from_sem

class Speaker:
    def generate(self, sem: dict):
        return code_from_sem(sem), explain_from_sem(sem)
