"""
Quick test script to verify model loading and basic inference
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def test_model(model_path):
    """Test if model loads and can generate text"""
    print(f"🧪 Testing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    try:
        # Load model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Test generation
        test_prompt = "Article 21 of the Indian Constitution"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🎯 Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    print("🇮🇳 LawGPT Model Testing")
    print("=" * 50)
    
    # Test pretrained model
    pretrained_path = "./checkpoints/lawgpt-pretrained"
    if os.path.exists(pretrained_path):
        test_model(pretrained_path)
    else:
        print(f"⏳ Pretrained model not yet available: {pretrained_path}")
    
    print()
    
    # Test fine-tuned model
    finetuned_path = "./checkpoints/lawgpt-finetuned"
    if os.path.exists(finetuned_path):
        test_model(finetuned_path)
    else:
        print(f"⏳ Fine-tuned model not yet available: {finetuned_path}")
