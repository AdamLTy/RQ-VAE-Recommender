from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    if isinstance(batch, SeqBatch):
        return batch
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 6:
            # Handle case where DataLoader returns a list/tuple of 6 tensors
            return SeqBatch(*batch)
        elif len(batch) == 1 and isinstance(batch[0], SeqBatch):
            # Handle case where collate_fn returns single SeqBatch in a list
            return batch[0]
        else:
            # Handle other list formats - try to convert first element
            if hasattr(batch[0], 'ids'):
                return batch[0]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)} with length {len(batch)}")
    else:
        # Fallback: assume it's already a proper batch
        return batch


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
