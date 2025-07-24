from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    # Handle different batch formats from PaddlePaddle DataLoader
    if isinstance(batch, SeqBatch):
        # Move SeqBatch tensors to device if needed
        if device is not None and device == "gpu":
            return SeqBatch(
                user_ids=batch.user_ids.cuda() if hasattr(batch.user_ids, 'cuda') else batch.user_ids,
                ids=batch.ids.cuda() if hasattr(batch.ids, 'cuda') else batch.ids,
                ids_fut=batch.ids_fut.cuda() if hasattr(batch.ids_fut, 'cuda') else batch.ids_fut,
                x=batch.x.cuda() if hasattr(batch.x, 'cuda') else batch.x,
                x_fut=batch.x_fut.cuda() if hasattr(batch.x_fut, 'cuda') else batch.x_fut,
                seq_mask=batch.seq_mask.cuda() if hasattr(batch.seq_mask, 'cuda') else batch.seq_mask
            )
        return batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 6:
        # Handle case where DataLoader returns a list/tuple of tensors
        tensors = batch
        if device is not None and device == "gpu":
            tensors = [t.cuda() if hasattr(t, 'cuda') else t for t in batch]
        return SeqBatch(*tensors)
    else:
        # Fallback: assume it's already a proper batch
        return batch


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
