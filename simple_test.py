"""
Simple interactive test for LawGPT
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def test_models():
    """Test both pretrained and fine-tuned models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")
    
    models_to_test = [
        ("./checkpoints/lawgpt-pretrained", "Pretrained"),
        ("./checkpoints/lawgpt-finetuned", "Fine-tuned")
    ]
    
    test_prompts = [
        "Article 21 of the Indian Constitution",
        "The Preamble states",
        "Right to equality means"
    ]
    
    for model_path, model_name in models_to_test:
        if not os.path.exists(model_path):
            print(f"❌ {model_name} model not found at {model_path}")
            continue
            
        print(f"\n🧪 Testing {model_name} Model")
        print("=" * 50)
        
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            model.to(device)
            
            for prompt in test_prompts:
                print(f"\n📝 Prompt: {prompt}")
                print("-" * 30)
                
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,  # Generate 50 more tokens
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=2
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                continuation = generated_text[len(prompt):].strip()
                print(f"🤖 Response: {continuation}")
                
        except Exception as e:
            print(f"❌ Error with {model_name} model: {str(e)}")

if __name__ == "__main__":
    test_models()
