"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=pad_idx)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    from tqdm import tqdm
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_confidence = 0.0
    total_tokens = 0
    
    for src, tgt in tqdm(data_iter, desc=f"Epoch {epoch_num} {'Train' if is_train else 'Eval'}"):
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        
        src_mask = make_src_mask(src, pad_idx=1).to(device)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=1).to(device)
        
        if is_train:
            optimizer.zero_grad()
            
        out = model.forward(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))
        
        if is_train:
            loss.backward()
            
            # --- Section 2.2 Explicit Gradient Logging ---
            # Log the L2 norm of the gradients for Query and Key weights of the first encoder layer
            import wandb
            if wandb.run is not None:
                try:
                    q_grad = model.encoder.layers[0].self_attn.w_q.weight.grad
                    k_grad = model.encoder.layers[0].self_attn.w_k.weight.grad
                    if q_grad is not None and k_grad is not None:
                        wandb.log({
                            "grad_norm/Q_weight": q_grad.norm(2).item(),
                            "grad_norm/K_weight": k_grad.norm(2).item(),
                            "global_step": wandb.run.step
                        })
                except Exception:
                    pass
            # ---------------------------------------------
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
        tokens = (tgt_y != 1).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        
        # Calculate prediction confidence for W&B Report Section 2.5
        with torch.no_grad():
            probs = torch.nn.functional.softmax(out, dim=-1)
            correct_probs = probs.gather(-1, tgt_y.unsqueeze(-1)).squeeze(-1)
            confidence = correct_probs[tgt_y != 1].sum().item()
            total_confidence += confidence
        
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_conf = total_confidence / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_conf


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos> (default 3).
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
    
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        out = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
        if next_word == end_symbol:
            break
            
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()
    predictions = []
    references = []
    idx2word = {idx: word for word, idx in tgt_vocab.items()}
    sos_idx = tgt_vocab.get('<sos>', 2)
    eos_idx = tgt_vocab.get('<eos>', 3)
    pad_idx = tgt_vocab.get('<pad>', 1)
    
    from tqdm import tqdm
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU"):
            for i in range(src.size(0)):
                s = src[i:i+1].to(device)
                t = tgt[i].tolist()
                sm = make_src_mask(s, pad_idx=pad_idx).to(device)
                
                ys = greedy_decode(model, s, sm, max_len, sos_idx, eos_idx, device)
                
                pred_words = [idx2word.get(idx, '<unk>') for idx in ys[0].tolist() if idx not in [sos_idx, eos_idx, pad_idx]]
                ref_words = [idx2word.get(idx, '<unk>') for idx in t if idx not in [sos_idx, eos_idx, pad_idx]]
                
                predictions.append(" ".join(pred_words))
                references.append([" ".join(ref_words)])
                
    try:
        from bleu import list_bleu
        return list_bleu(references, predictions) * 100
    except ImportError:
        try:
            from nltk.translate.bleu_score import corpus_bleu
            refs = [[ref[0].split()] for ref in references]
            preds = [pred.split() for pred in predictions]
            return corpus_bleu(refs, preds) * 100
        except:
            return 0.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': {
            'src_vocab_size': len(model.src_vocab),
            'tgt_vocab_size': len(model.tgt_vocab),
        }
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0)


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import wandb
    import argparse
    from dataset import get_dataloaders
    from lr_scheduler import NoamScheduler
    
    parser = argparse.ArgumentParser(description="Train Transformer NMT")
    parser.add_argument("--scheduler", type=str, default="noam", choices=["noam", "fixed"], help="Type of learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--no_scale", action="store_true", help="Disable 1/sqrt(d_k) scaling (for Section 2.2)")
    parser.add_argument("--learned_pos", action="store_true", help="Use Learned Positional Embeddings (for Section 2.4)")
    parser.add_argument("--no_smoothing", action="store_true", help="Disable Label Smoothing (for Section 2.5)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dynamically name the wandb run based on the experiments
    if args.no_scale:
        run_name = "no_scale_experiment"
    elif args.learned_pos:
        run_name = "learned_pos_experiment"
    elif args.no_smoothing:
        run_name = "no_smoothing_experiment"
    else:
        run_name = "noam_scheduler_run" if args.scheduler == "noam" else "fixed_lr_1e-4_run"
        
    wandb.init(project="da6401-a3", name=run_name)
    
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=32)
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1
    ).to(device)
    
    import model as md
    if args.no_scale:
        md.USE_SCALE = False
    if args.learned_pos:
        md.USE_LEARNED_POS = True
        
    # Log gradients for Section 2.2
    wandb.watch(model, log="gradients", log_freq=10)
    
    if args.scheduler == "noam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    else:
        # Fixed Learning Rate Ablation Study (Section 2.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None
        
    smoothing_val = 0.0 if args.no_smoothing else 0.1
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1, smoothing=smoothing_val).to(device)
    
    num_epochs = args.epochs
    import math
    for epoch in range(num_epochs):
        train_loss, train_conf = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss, val_conf = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        
        train_perplexity = math.exp(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Conf {val_conf:.4f}, Val BLEU {val_bleu:.2f}, LR {current_lr:.6f}")
        wandb.log({
            "train/loss": train_loss, 
            "val/loss": val_loss, 
            "train/perplexity": train_perplexity,
            "val/bleu": val_bleu,
            "learning_rate": current_lr,
            "train_confidence": train_conf,
            "val_confidence": val_conf,
            "epoch": epoch
        })
        save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pt")
        
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({'test_bleu': bleu})
    print(f"Final Test BLEU: {bleu:.2f}")

if __name__ == "__main__":
    run_training_experiment()
