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
from data_preprocessing import prepare_datasets

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

def pretrain_gpt2():
    """Main pretraining function"""
    print("🚀 Starting GPT-2 Pretraining on Indian Constitution Data")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    print("📚 Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    print("📊 Preparing datasets...")
    pretraining_dataset, _ = prepare_datasets("COI.json")
    
    # Tokenize the dataset
    print("🔤 Tokenizing dataset...")
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
    output_dir = "./checkpoints/lawgpt-pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments optimized for RTX 3050
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Increased for RTX 3050
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
        warmup_steps=100,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision for RTX 3050
        dataloader_drop_last=True,
        logging_dir=f"{output_dir}/logs",
        report_to=[],  # Disable wandb logging
        seed=42,
        remove_unused_columns=False,
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        no_cuda=False,  # Explicitly enable CUDA
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🏋️ Starting pretraining...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Pretraining completed! Model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    pretrain_gpt2()
