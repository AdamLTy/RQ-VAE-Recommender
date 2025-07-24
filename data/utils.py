from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    # In PaddlePaddle, tensors are already on the correct device by default
    # but we can ensure they're on the right device if needed
    if isinstance(batch, SeqBatch):
        return batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 6:
        # Handle case where DataLoader returns a list/tuple of tensors
        return SeqBatch(*batch)
    else:
        # Fallback: assume it's already a proper batch
        return batch


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
