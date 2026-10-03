"""The tiny 'world' our agent lives in: a product catalog and order questions.

Every task has a known correct answer, so we can automatically label each
agent run as success or failure. That is what makes training possible later.
"""
import random

CATALOG = [
    {"name": "Widget",        "price": 19.99, "stock": 40},
    {"name": "Gadget",        "price": 34.50, "stock": 25},
    {"name": "USB Cable",     "price": 7.25,  "stock": 80},
    {"name": "Desk Lamp",     "price": 42.00, "stock": 18},
    {"name": "Notebook",      "price": 4.75,  "stock": 120},
    {"name": "Headphones",    "price": 89.90, "stock": 15},
    {"name": "Water Bottle",  "price": 16.40, "stock": 60},
    {"name": "Phone Stand",   "price": 12.30, "stock": 35},
    {"name": "Backpack",      "price": 54.80, "stock": 22},
    {"name": "Keyboard",      "price": 67.20, "stock": 14},
    {"name": "Mouse Pad",     "price": 9.60,  "stock": 70},
    {"name": "Webcam",        "price": 48.35, "stock": 16},
]


def make_tasks(n, seed=0):
    """Create n order questions, each with a known correct total."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        item = rng.choice(CATALOG)
        qty = rng.randint(2, 9)
        disc = rng.choice([5, 10, 15, 20, 25])
        question = f"How much do {qty} units of {item['name']} cost with a {disc}% discount?"
        expected = round(qty * item["price"] * (1 - disc / 100), 2)
        tasks.append({"question": question, "expected": expected})
    return tasks
