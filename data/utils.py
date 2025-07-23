from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    # In PaddlePaddle, tensors are already on the correct device by default
    # but we can ensure they're on the right device if needed
    return SeqBatch(*[v for _,v in batch._asdict().items()])


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
