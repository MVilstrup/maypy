
class Interpretation:
    def __init__(self, result, confidence=True):
        self.result = result
        self.confidence = confidence

    def __bool__(self):
        return bool(self.result and self.confidence)

    def __str__(self):
        confidence = "Conclusive" if self.confidence else "Inconclusive"
        return f"{confidence}({'Pass' if self.result else 'Failure'})"