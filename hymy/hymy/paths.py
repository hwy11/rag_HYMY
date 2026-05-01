from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def merged_markdown_path(batch_num: int) -> Path:
    return ROOT_DIR / f"merged_output_{batch_num}.md"


def processed_json_path(batch_num: int) -> Path:
    return OUTPUT_DIR / f"processed_data_{batch_num}.json"


def enriched_json_path(batch_num: int) -> Path:
    return OUTPUT_DIR / f"processed_data_{batch_num}_enriched.json"


def final_markdown_path(batch_num: int) -> Path:
    return OUTPUT_DIR / f"final_output_{batch_num}.md"

