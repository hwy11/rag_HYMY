from hymy.paths import final_markdown_path
from hymy.pipeline import iter_batches, verify_batch


def verify_and_finalize():
    for batch_num in iter_batches():
        verification = verify_batch(batch_num)
        if verification is None:
            print(f"Missing {final_markdown_path(batch_num)}")
            continue

        entry_count, answer_tag_count = verification
        print(
            f"File {final_markdown_path(batch_num)}: "
            f"Entries={entry_count}, Answers={answer_tag_count}"
        )

if __name__ == "__main__":
    verify_and_finalize()
