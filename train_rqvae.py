import gin
import os
import paddle
import numpy as np
import swanlab

from data.processed import RecDataset
from data.h5_dataset import H5PretrainedDataset
from data.schemas import SeqBatch
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from paddle.optimizer import AdamW
from tqdm import tqdm


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    amp=False,
    swanlab_logging=False,
    do_eval=True,
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=256,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    # H5 dataset parameters
    h5_item_data_path="data/preprocessed/item_data.h5",
    h5_test_ratio=0.2
):
    if swanlab_logging:
        params = locals()

    # Set device for PaddlePaddle
    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    paddle.device.set_device(device)

    print("Using H5 pretrained dataset")
    # Create dataset object for kmeans initialization and evaluation
    train_dataset = H5PretrainedDataset(
        item_data_path=h5_item_data_path,
        train_test_split="train" if do_eval else "all",
        test_ratio=h5_test_ratio
    )
    
    # Define collate function for batching
    def collate_fn(batch):
        if len(batch) == 1:
            return batch[0]
        # Concatenate item-level tensors
        user_ids = paddle.concat([item.user_ids for item in batch], axis=0)
        ids = paddle.concat([item.ids for item in batch], axis=0)
        ids_fut = paddle.concat([item.ids_fut for item in batch], axis=0)
        x = paddle.concat([item.x for item in batch], axis=0)
        x_fut = paddle.concat([item.x_fut for item in batch], axis=0)
        seq_mask = paddle.concat([item.seq_mask for item in batch], axis=0)
        return SeqBatch(user_ids=user_ids, ids=ids, ids_fut=ids_fut, x=x, x_fut=x_fut, seq_mask=seq_mask)
    
    # Create train dataloader with shuffle (equivalent to RandomSampler)
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        use_shared_memory=False,
        prefetch_factor=2
    )
    train_dataloader = cycle(train_dataloader)
    
    if do_eval:
        eval_dataset = H5PretrainedDataset(
            item_data_path=h5_item_data_path,
            train_test_split="eval",
            test_ratio=h5_test_ratio
        )
        # Create eval dataloader without shuffle (sequential)
        eval_dataloader = paddle.io.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            use_shared_memory=False,
            prefetch_factor=2
        )
    
    # For H5 dataset, we need to get embedding dimension from the data
    import h5py
    with h5py.File(h5_item_data_path, 'r') as f:
        h5_embedding_dim = f.attrs['embedding_dim']
    vae_input_dim = h5_embedding_dim
    print(f"H5 dataset embedding dimension: {h5_embedding_dim}")
    
    # Use train_dataset for evaluation (same as original logic)
    index_dataset = train_dataset

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )

    optimizer = AdamW(
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    if swanlab_logging:
        swanlab.init(
            project="rq-vae-training",
            config=params
        )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = paddle.load(pretrained_rqvae_path)
        optimizer.set_state_dict(state["optimizer"])
        start_iter = state["iter"]+1

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer.rq_vae = model

    with tqdm(initial=start_iter, total=start_iter+iterations) as pbar:
        losses = [[], [], []]
        for iter in range(start_iter, start_iter+1+iterations):
            model.train()
            total_loss = 0
            t = 0.2
            if iter == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(train_dataset[paddle.arange(min(20000, len(train_dataset)))], device)
                model(kmeans_init_data, t)

            optimizer.clear_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                if amp:
                    with paddle.amp.auto_cast():
                        model_output = model(data, gumbel_t=t)
                        loss = model_output.loss
                        loss = loss / gradient_accumulate_every
                        total_loss += loss
                else:
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            total_loss.backward()

            losses[0].append(total_loss.item())
            losses[1].append(model_output.reconstruction_loss.item())
            losses[2].append(model_output.rqvae_loss.item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

            optimizer.step()

            id_diversity_log = {}
            if swanlab_logging:
                # Compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].item() for i in range(vae_n_layers)
                }
                train_log = {
                    "learning_rate": optimizer.get_lr(),
                    "total_loss": total_loss.item(),
                    "reconstruction_loss": model_output.reconstruction_loss.item(),
                    "rqvae_loss": model_output.rqvae_loss.item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.item(),
                    **emb_norms_avg_log,
                }

            if do_eval and ((iter+1) % eval_every == 0 or iter+1 == iterations):
                print(f"[DEBUG] Starting evaluation at iteration {iter+1}")
                print(f"[DEBUG] Switching model to eval mode")
                model.eval()
                print(f"[DEBUG] Creating eval dataloader tqdm, eval_dataset size: {len(eval_dataset)}")
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=True) as pbar_eval:
                    eval_losses = [[], [], []]
                    print(f"[DEBUG] Starting batch iteration over eval_dataloader")
                    batch_count = 0
                    for batch in pbar_eval:
                        batch_count += 1
                        if batch_count % 10 == 1:  # Print every 10 batches
                            print(f"[DEBUG] Processing eval batch {batch_count}")
                        data = batch_to(batch, device)
                        with paddle.no_grad():
                            if batch_count % 10 == 1:
                                print(f"[DEBUG] Running model inference for batch {batch_count}")
                            eval_model_output = model(data, gumbel_t=t)
                            if batch_count % 10 == 1:
                                print(f"[DEBUG] Model inference completed for batch {batch_count}")

                        eval_losses[0].append(eval_model_output.loss.item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.item())
                    
                    print(f"[DEBUG] Completed all eval batches, total batches processed: {batch_count}")
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    print(f"[DEBUG] Computed eval losses: total={eval_losses[0]:.4f}, recon={eval_losses[1]:.4f}, rqvae={eval_losses[2]:.4f}")
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    print(f"[DEBUG] Evaluation phase completed")
                    
            if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                state = {
                    "iter": iter,
                    "model": model.state_dict(),
                    "model_config": model.config,
                    "optimizer": optimizer.state_dict()
                }

                if not os.path.exists(save_dir_root):
                    os.makedirs(save_dir_root)

                paddle.save(state, save_dir_root + f"checkpoint_{iter}.pdparams")
                
            if (iter+1) % eval_every == 0 or iter+1 == iterations:
                print(f"[DEBUG] Starting tokenizer operations at iteration {iter+1}")
                tokenizer.reset()
                print(f"[DEBUG] Tokenizer reset completed")
                model.eval()
                print(f"[DEBUG] Starting precompute_corpus_ids, index_dataset size: {len(index_dataset)}")
                corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                print(f"[DEBUG] Precompute_corpus_ids completed, corpus_ids shape: {corpus_ids.shape}")
                print(f"[DEBUG] Computing max_duplicates")
                max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                print(f"[DEBUG] Max_duplicates computed: {max_duplicates.item():.4f}")
                
                print(f"[DEBUG] Computing RQVAE entropy")
                _, counts = paddle.unique(corpus_ids[:,:-1], axis=0, return_counts=True)
                p = counts / corpus_ids.shape[0]
                rqvae_entropy = -(p*paddle.log(p)).sum()
                print(f"[DEBUG] RQVAE entropy computed: {rqvae_entropy.item():.4f}")

                print(f"[DEBUG] Computing codebook usage for {vae_n_layers} layers")
                for cid in range(vae_n_layers):
                    print(f"[DEBUG] Computing codebook usage for layer {cid}")
                    _, counts = paddle.unique(corpus_ids[:,cid], return_counts=True)
                    usage = len(counts) / vae_codebook_size
                    id_diversity_log[f"codebook_usage_{cid}"] = usage
                    print(f"[DEBUG] Layer {cid} codebook usage: {usage:.4f}")

                print(f"[DEBUG] Preparing id_diversity_log for logging")
                id_diversity_log["rqvae_entropy"] = rqvae_entropy.item()
                id_diversity_log["max_id_duplicates"] = max_duplicates.item()
                print(f"[DEBUG] Id_diversity_log prepared with {len(id_diversity_log)} entries")
            
            if swanlab_logging:
                print(f"[DEBUG] Starting swanlab logging at iteration {iter+1}")
                log_data = {
                    **train_log,
                    **id_diversity_log
                }
                print(f"[DEBUG] Prepared log data with {len(log_data)} entries")
                swanlab.log(log_data)
                print(f"[DEBUG] Swanlab logging completed successfully")

            print(f"[DEBUG] Updating progress bar for iteration {iter+1}")
            pbar.update(1)
            print(f"[DEBUG] Iteration {iter+1} completed successfully")
    
    if swanlab_logging:
        swanlab.finish()


if __name__ == "__main__":
    parse_config()
    train()
