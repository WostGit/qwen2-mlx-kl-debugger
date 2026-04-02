"""Shared prompt set and helpers."""

PROMPTS = [
    "The capital of France is",
    "In one sentence, gravity means",
    "Python is a programming language that",
    "A healthy breakfast often includes",
    "The opposite of hot is",
    "The largest planet in our solar system is",
    "When water freezes, it becomes",
    "A synonym for happy is",
    "The author wrote a book about",
    "Machine learning can help with",
    "The color of a clear daytime sky is",
    "A triangle has how many sides?",
    "The chemical symbol for water is",
    "An electric car uses",
    "To boil pasta, first",
    "The moon orbits",
    "A laptop typically has",
    "The fastest land animal is",
    "Photosynthesis happens in",
    "A map helps you",
]


def budgeted_prompts(budget: int) -> list[tuple[int, str]]:
    rows = []
    n = len(PROMPTS)
    for i in range(int(budget)):
        pid = i % n
        rows.append((pid, PROMPTS[pid]))
    return rows
