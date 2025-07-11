"""
Enhanced fine-tuning script for multiple legal datasets
Optimized for RTX 3050 with large datasets
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

def format_instruction_data(examples, tokenizer, max_length=512):
    """Format instruction data for supervised fine-tuning"""
    formatted_texts = []
    
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Format: Instruction: <instruction>\nAnswer: <output><eos>
        if input_text.strip():
            formatted_text = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: {output}{tokenizer.eos_token}"
        else:
            formatted_text = f"Instruction: {instruction}\nAnswer: {output}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    
    return tokenized

def enhanced_finetune_gpt2(pretrained_model_path=None, use_enhanced_data=True):
    """Enhanced fine-tuning function with larger datasets"""
    print("🚀 Starting Enhanced GPT-2 Fine-tuning on Multiple Legal Datasets")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and tokenizer
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"📚 Loading pretrained model from {pretrained_model_path}...")
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path)
    else:
        print("📚 Loading base GPT-2 model and tokenizer...")
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare enhanced datasets
    print("📊 Preparing enhanced instruction dataset...")
    if use_enhanced_data:
        _, instruction_dataset = prepare_enhanced_datasets()
    else:
        from data_preprocessing import prepare_datasets
        _, instruction_dataset = prepare_datasets("COI.json")
    
    print(f"📈 Total instruction samples: {len(instruction_dataset)}")
    
    # Tokenize the dataset
    print("🔤 Tokenizing enhanced instruction dataset...")
    tokenized_dataset = instruction_dataset.map(
        lambda x: format_instruction_data(x, tokenizer),
        batched=True,
        remove_columns=instruction_dataset.column_names,
        desc="Tokenizing instruction data"
    )
    
    # Split into train and eval (90-10 split for large dataset)
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
    
    print(f"📈 Train dataset size: {len(train_dataset)}")
    print(f"📉 Eval dataset size: {len(eval_dataset)}")
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is not a masked language model
    )
    
    # Create output directory
    output_dir = "./checkpoints/lawgpt-enhanced-finetuned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate training steps
    batch_size = 4  # Conservative for RTX 3050
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_epochs = 2  # Reduced epochs for large dataset
    total_steps = math.ceil(len(train_dataset) / effective_batch_size) * num_epochs
    
    print(f"🎯 Training configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps}")
    
    # Training arguments optimized for RTX 3050 with large dataset
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=min(500, total_steps // 10),  # 10% warmup
        logging_steps=50,
        save_steps=min(500, total_steps // 4),  # Save 4 times during training
        eval_steps=min(500, total_steps // 4),
        save_total_limit=3,
        eval_strategy="steps",
        fp16=True,  # Enable mixed precision for RTX 3050
        dataloader_drop_last=True,
        logging_dir=f"{output_dir}/logs",
        report_to=[],  # Disable wandb logging
        seed=42,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        max_grad_norm=1.0,
        no_cuda=False,
        dataloader_num_workers=0,  # Prevent multiprocessing issues
        max_steps=total_steps,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Start training
    print("🏋️ Starting enhanced fine-tuning...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving enhanced model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Enhanced fine-tuning completed! Model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    # Use pretrained model if available
    pretrained_path = "./checkpoints/lawgpt-pretrained"
    enhanced_finetune_gpt2(
        pretrained_path if os.path.exists(pretrained_path) else None,
        use_enhanced_data=True
    )
