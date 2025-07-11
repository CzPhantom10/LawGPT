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

def format_instruction_data(examples, tokenizer, max_length=512):
    """Format instruction data for supervised fine-tuning"""
    formatted_texts = []
    
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Format: <instruction>\n<input>\n<output><eos>
        if input_text.strip():
            formatted_text = f"{instruction}\nInput: {input_text}\nOutput: {output}{tokenizer.eos_token}"
        else:
            formatted_text = f"{instruction}\nOutput: {output}{tokenizer.eos_token}"
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

def finetune_gpt2(pretrained_model_path=None):
    """Main fine-tuning function"""
    print("🎯 Starting GPT-2 Fine-tuning on Indian Constitution Instructions")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # Prepare datasets
    print("📊 Preparing instruction dataset...")
    _, instruction_dataset = prepare_datasets("COI.json")
    
    # Tokenize the dataset
    print("🔤 Tokenizing instruction dataset...")
    tokenized_dataset = instruction_dataset.map(
        lambda x: format_instruction_data(x, tokenizer),
        batched=True,
        remove_columns=instruction_dataset.column_names,
        desc="Tokenizing instruction data"
    )
    
    # Split into train and eval (80-20 split)
    train_size = int(0.8 * len(tokenized_dataset))
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
    output_dir = "./checkpoints/lawgpt-finetuned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments optimized for RTX 3050
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=6,  # Increased for RTX 3050
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=2,  # Effective batch size = 6 * 2 = 12
        warmup_steps=50,
        logging_steps=5,
        save_steps=25,
        eval_steps=25,
        save_total_limit=2,
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
        learning_rate=3e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        no_cuda=False,  # Explicitly enable CUDA
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
    
    # Start training
    print("🏋️ Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Fine-tuning completed! Model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    # Try to use pretrained model if available
    pretrained_path = "./checkpoints/lawgpt-pretrained"
    finetune_gpt2(pretrained_path if os.path.exists(pretrained_path) else None)
