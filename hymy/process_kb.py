from hymy.pipeline import iter_batches, process_batch


def main():
    for batch_num in iter_batches():
        processed_count = process_batch(batch_num)
        if processed_count:
            print(f"Processed batch {batch_num}: {processed_count} entries")

if __name__ == "__main__":
    main()
