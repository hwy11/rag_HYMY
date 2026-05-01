from hymy.pipeline import enrich_batch, iter_batches

if __name__ == "__main__":
    for batch_num in iter_batches():
        enriched_count = enrich_batch(batch_num)
        if enriched_count:
            print(f"Enriched batch {batch_num}: {enriched_count} entries")
