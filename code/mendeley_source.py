"""Read source reviews and sentiment labels from the Mendeley archive."""
import ast
from pathlib import Path


def load_reviews(root: Path):
    records = []
    for film_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        polarity_text = (film_dir / "polarity.txt").read_text(encoding="utf-8")
        polarities = ast.literal_eval(polarity_text.split("=", 1)[1].strip())
        for index, polarity in enumerate(polarities, start=1):
            path = film_dir / f"{index}.txt"
            text = path.read_text(encoding="utf-8").strip()
            if text:
                records.append({
                    "id": f"mendeley_{film_dir.name}_{index}",
                    "movie": film_dir.name,
                    "text": text,
                    "sentiment": "positive" if int(polarity) == 1 else "negative",
                })
    return records
