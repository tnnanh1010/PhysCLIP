"""
CLIP Fine-tuning for Astronomy Images - Standard Version
Uses in-batch negatives for contrastive learning
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

from transformers import (
    VisionTextDualEncoderModel,
    AutoTokenizer,
    CLIPImageProcessor,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb


class AstronomyImageCaptionDataset(Dataset):
    """Dataset for astronomy image-caption pairs."""
    
    def __init__(
        self,
        jsonl_path: str,
        image_processor: CLIPImageProcessor,
        tokenizer: AutoTokenizer,
        max_length: int = 77,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load JSONL data
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('image_path') and item.get('description'):
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} image-caption pairs from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image (minimal augmentation for scientific accuracy)
        try:
            image = Image.open(item['image_path']).convert('RGB')
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            pixel_values = torch.zeros(3, 224, 224)
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            item['description'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs.input_ids.squeeze(0),
            'attention_mask': text_inputs.attention_mask.squeeze(0),
        }


def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Standard CLIP contrastive loss using in-batch negatives.
    
    For each image-text pair in batch:
    - Positive: the matching text/image
    - Negatives: all other texts/images in the batch
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix (batch_size x batch_size)
    logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
    
    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    # Symmetric loss: image-to-text and text-to-image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    
    # Compute accuracy
    with torch.no_grad():
        i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
        t2i_acc = (logits.t().argmax(dim=1) == labels).float().mean()
        accuracy = (i2t_acc + t2i_acc) / 2
    
    return loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, accelerator):
    """Evaluate model on validation set."""
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        outputs = model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True
        )
        
        image_embeds = accelerator.gather(outputs.image_embeds)
        text_embeds = accelerator.gather(outputs.text_embeds)
        
        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Normalize
    all_image_embeds = F.normalize(all_image_embeds, dim=-1)
    all_text_embeds = F.normalize(all_text_embeds, dim=-1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_embeds, all_text_embeds.t())
    
    # Compute recall@k
    metrics = {}
    for k in [1, 5, 10]:
        # Image-to-text
        topk = similarity_matrix.topk(k, dim=1).indices
        correct = (topk == torch.arange(len(similarity_matrix)).unsqueeze(1)).any(dim=1)
        metrics[f'i2t_recall@{k}'] = correct.float().mean().item()
        
        # Text-to-image
        topk = similarity_matrix.t().topk(k, dim=1).indices
        correct = (topk == torch.arange(len(similarity_matrix)).unsqueeze(1)).any(dim=1)
        metrics[f't2i_recall@{k}'] = correct.float().mean().item()
    
    model.train()
    return metrics


def train(args):
    """Main training function."""
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
    )
    
    set_seed(args.seed)
    
    if accelerator.is_local_main_process and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    
    # Load processors
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    
    # Create datasets
    train_dataset = AstronomyImageCaptionDataset(
        jsonl_path=args.train_data,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    if args.val_data:
        val_dataset = AstronomyImageCaptionDataset(
            jsonl_path=args.val_data,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        # Use 5% for validation
        val_size = int(0.05 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Initialize model
    if accelerator.is_local_main_process:
        print(f"Loading vision: {args.vision_model}, text: {args.text_model}")
    
    model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=args.text_model,
        projection_dim=args.projection_dim,
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # PHASE 1: Warmup with frozen encoders
    if args.warmup_epochs > 0:
        if accelerator.is_local_main_process:
            print(f"\n{'='*60}\nPHASE 1: Warmup projection layers\n{'='*60}\n")
        
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.text_model.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )
        
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        
        for epoch in range(args.warmup_epochs):
            model.train()
            total_loss = 0
            total_acc = 0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Warmup {epoch+1}/{args.warmup_epochs}",
                disable=not accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                with accelerator.accumulate(model):
                    outputs = model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        return_dict=True
                    )
                    
                    loss, acc = contrastive_loss(
                        outputs.image_embeds,
                        outputs.text_embeds,
                        temperature=args.temperature
                    )
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    total_acc += acc.item()
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})
            
            if accelerator.is_local_main_process:
                print(f"Warmup Epoch {epoch+1}: Loss={total_loss/len(train_dataloader):.4f}")
        
        model = accelerator.unwrap_model(model)
    
    # PHASE 2: Full fine-tuning
    if accelerator.is_local_main_process:
        print(f"\n{'='*60}\nPHASE 2: Full fine-tuning\n{'='*60}\n")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )
    
    best_recall = 0.0
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True
                )
                
                loss, acc = contrastive_loss(
                    outputs.image_embeds,
                    outputs.text_embeds,
                    temperature=args.temperature
                )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                total_acc += acc.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Evaluate
        val_metrics = evaluate(model, val_dataloader, accelerator)
        
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch+1}: Loss={total_loss/len(train_dataloader):.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': total_loss / len(train_dataloader),
                    'train/accuracy': total_acc / len(train_dataloader),
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })
            
            # Save best model
            avg_recall = (val_metrics['i2t_recall@1'] + val_metrics['t2i_recall@1']) / 2
            if avg_recall > best_recall:
                best_recall = avg_recall
                output_dir = Path(args.output_dir) / "best_model"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir)
                image_processor.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                print(f"Saved best model (recall@1={avg_recall:.4f})")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % args.save_every == 0:
                output_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir)
                image_processor.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
    
    if accelerator.is_local_main_process:
        print("\nTraining complete!")
        if args.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for astronomy images")
    
    # Model
    parser.add_argument("--vision_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_model", default="arnosimons/astro-hep-bert")
    parser.add_argument("--projection_dim", type=int, default=512)
    
    # Data
    parser.add_argument("--train_data", required=True, help="Path to training JSONL file")
    parser.add_argument("--val_data", default=None, help="Path to validation JSONL file")
    parser.add_argument("--max_length", type=int, default=77)
    
    # Training
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Other
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="astronomy-clip")
    parser.add_argument("--run_name", default="clip-astro-standard")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()