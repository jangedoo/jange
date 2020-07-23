class ClassificationResult:
    def __init__(self, label, proba, raw):
        self.label = label
        self.proba = proba
        self.raw = raw

    def __repr__(self):
        return f"ClassificationResult(label={self.label}, proba={self.proba})"
