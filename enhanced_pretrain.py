"""
Enhanced pretraining script for multiple legal datasets
"""
import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from enhanced_preprocessing import prepare_enhanced_datasets
import math

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the texts for pretraining"""
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding=False, 
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True
    )
    
    # Remove overflow tokens that are too short
    input_batch = []
    for length, input_ids in zip(tokenized["length"], tokenized["input_ids"]):
        if length > 10:  # Keep only sequences longer than 10 tokens
            input_batch.append(input_ids)
    
    return {"input_ids": input_batch}

def enhanced_pretrain_gpt2():
    """Enhanced pretraining function with multiple legal datasets"""
    print("🚀 Starting Enhanced GPT-2 Pretraining on Multiple Legal Datasets")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and tokenizer
    print("📚 Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare enhanced datasets
    print("📊 Preparing enhanced datasets...")
    pretraining_dataset, _ = prepare_enhanced_datasets()
    
    print(f"📈 Total pretraining samples: {len(pretraining_dataset)}")
    
    # Tokenize the dataset
    print("🔤 Tokenizing enhanced dataset...")
    tokenized_dataset = pretraining_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=pretraining_dataset.column_names,
        desc="Tokenizing data"
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is not a masked language model
    )
    
    # Create output directory
    output_dir = "./checkpoints/lawgpt-enhanced-pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate training parameters
    batch_size = 6  # Optimized for RTX 3050
    gradient_accumulation_steps = 2
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_epochs = 3
    total_steps = math.ceil(len(tokenized_dataset) / effective_batch_size) * num_epochs
    
    print(f"🎯 Training configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps}")
    
    # Training arguments optimized for RTX 3050
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=min(200, total_steps // 10),
        logging_steps=20,
        save_steps=min(200, total_steps // 5),
        save_total_limit=3,
        fp16=True,  # Enable mixed precision for RTX 3050
        dataloader_drop_last=True,
        logging_dir=f"{output_dir}/logs",
        report_to=[],  # Disable wandb logging
        seed=42,
        remove_unused_columns=False,
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        no_cuda=False,
        dataloader_num_workers=0,
        prediction_loss_only=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Start training
    print("🏋️ Starting enhanced pretraining...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving enhanced pretrained model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Enhanced pretraining completed! Model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    enhanced_pretrain_gpt2()
