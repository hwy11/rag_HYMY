from hymy.pipeline import iter_batches, reconstruct_batch


def main():
    for batch_num in iter_batches():
        reconstructed_count = reconstruct_batch(batch_num)
        if reconstructed_count:
            print(f"Reconstructed batch {batch_num}: {reconstructed_count} entries")

if __name__ == "__main__":
    main()
