import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-7):
    def lr_lambda(current_step):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / optimizer.defaults['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, model_name, train_loader, val_loader, device,
                num_epochs=30, lr=5e-5, weight_decay=1e-4, warmup_epochs=2,
                grad_clip=1.0, resume_if_exists=True, use_amp=True):
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    save_dir = f"checkpoints_{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, "latest.pth")
    best_path = os.path.join(save_dir, "best.pth")

    start_epoch = 1
    best_val_acc = 0.0
    log = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Resuming logic
    if resume_if_exists and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        try: 
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except: 
            pass
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        log = ckpt["log"]
        print(f"[{model_name}] Resuming from epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"{model_name} Train Epoch {epoch}")

        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            train_bar.set_postfix(avg_loss=f"{train_loss / total:.4f}")

        train_loss /= total
        train_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"{model_name} Valid Epoch {epoch}")
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_bar.set_postfix(avg_loss=f"{val_loss / val_total:.4f}")

        val_loss /= val_total
        val_acc = val_correct / val_total

        # --- Logging ---
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["train_acc"].append(train_acc)
        log["val_acc"].append(val_acc)

        print(f"\n{model_name} Epoch {epoch}/{num_epochs} - "
              f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        # --- Checkpoint Saving ---
        save_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "log": log
        }

        # Save latest
        torch.save(save_dict, latest_path)

        # Save best if current accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict["best_val_acc"] = best_val_acc
            torch.save(save_dict, best_path)

        # Periodic save every 5 epochs
        if epoch % 5 == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
            torch.save(save_dict, epoch_path)

        # Save logs to JSON after each epoch
        with open(os.path.join(save_dir, "logs.json"), "w") as f:
            json.dump(log, f, indent=2)