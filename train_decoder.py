import argparse
import os
import gin
import paddle
import swanlab

from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.h5_dataset import H5SequenceDataset, create_h5_sequence_dataloader
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
# from huggingface_hub import login
from paddle.optimizer import AdamW
from paddle.io import DataLoader
from tqdm import tqdm


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    swanlab_logging=False,
    force_dataset_process=False,
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=256,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    data_path=None,
    # H5 dataset parameters  
    h5_sequence_data_path="data/preprocessed/sequence_data.h5",
    h5_item_data_path="data/preprocessed/item_data.h5",
    h5_max_seq_len=200,
    # Corpus IDs caching
    corpus_ids_cache_path="cache/corpus_ids.pkl"
):  

    if swanlab_logging:
        params = locals()

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    paddle.device.set_device(device)

    if swanlab_logging:
        run = swanlab.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    # Use H5 sequence dataset for decoder training
    print("Using H5 sequence dataset for decoder training")
    
    # Create sequence datasets for training and evaluation
    train_dataset = H5SequenceDataset(
        sequence_data_path=h5_sequence_data_path,
        item_data_path=h5_item_data_path,
        is_train=True,
        max_seq_len=h5_max_seq_len,
        subsample=train_data_subsample
    )
    
    eval_dataset = H5SequenceDataset(
        sequence_data_path=h5_sequence_data_path,
        item_data_path=h5_item_data_path,
        is_train=False, 
        max_seq_len=h5_max_seq_len,
        subsample=False
    )
    
    # For item_dataset, use train_dataset (compatible with existing code)
    item_dataset = train_dataset
    
    # Create sequence dataloaders
    train_dataloader = create_h5_sequence_dataloader(
        sequence_data_path=h5_sequence_data_path,
        item_data_path=h5_item_data_path,
        batch_size=batch_size,
        is_train=True,
        max_seq_len=h5_max_seq_len,
        subsample=train_data_subsample,
        shuffle=True
    )
    
    eval_dataloader = create_h5_sequence_dataloader(
        sequence_data_path=h5_sequence_data_path,
        item_data_path=h5_item_data_path,
        batch_size=batch_size,
        is_train=False,
        max_seq_len=h5_max_seq_len,
        subsample=False,
        shuffle=True
    )
    
    train_dataloader = cycle(train_dataloader)
    

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
    tokenizer.precompute_corpus_ids(item_dataset, cache_path=corpus_ids_cache_path)
    
    # HuggingFace functionality removed
    # if push_vae_to_hf:
    #     login()
    #     tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode
    )

    lr_scheduler = InverseSquareRootScheduler(
        learning_rate=learning_rate,
        warmup_steps=10000
    )
    
    optimizer = AdamW(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=weight_decay
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = paddle.load(pretrained_decoder_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1


    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=False) as pbar:
        for step in range(iterations):
            model.train()
            total_loss = 0
            optimizer.clear_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                model_output = model(tokenized_data)
                loss = model_output.loss / gradient_accumulate_every
                total_loss += loss
                
                if swanlab_logging:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                total_loss.backward()
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            optimizer.step()
            lr_scheduler.step()

            if (step+1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                eval_debug_metrics = {}
                eval_losses = []
                
                # Limit partial evaluation to a few batches for efficiency
                eval_batch_count = 0
                max_eval_batches = 10
                
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with paddle.no_grad():
                        model_output_eval = model(tokenized_data)
                        eval_losses.append(model_output_eval.loss.detach().cpu().item())

                    if swanlab_logging:
                        batch_eval_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        # Accumulate metrics
                        for key, value in batch_eval_metrics.items():
                            if key not in eval_debug_metrics:
                                eval_debug_metrics[key] = []
                            eval_debug_metrics[key].append(value)
                    
                    eval_batch_count += 1
                    if eval_batch_count >= max_eval_batches:
                        break
                
                # Average the metrics
                if swanlab_logging and eval_debug_metrics:
                    for key in eval_debug_metrics:
                        eval_debug_metrics[key] = sum(eval_debug_metrics[key]) / len(eval_debug_metrics[key])
                    eval_debug_metrics["eval_loss"] = sum(eval_losses) / len(eval_losses)
                    swanlab.log(eval_debug_metrics)

            if (step+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                with tqdm(eval_dataloader, desc=f'Eval {step+1}', disable=False) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if swanlab_logging:
                    swanlab.log(eval_metrics)
                
                metrics_accumulator.reset()

            if True:
                if (step+1) % save_model_every == 0 or step+1 == iterations:
                    state = {
                        "iter": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    paddle.save(state, save_dir_root + f"checkpoint_{step}.pdparams")
                
                if swanlab_logging:
                    swanlab.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if swanlab_logging:
        swanlab.finish()


if __name__ == "__main__":
    parse_config()
    train()
