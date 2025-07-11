"""
Model Comparison Script
Compare performance between original and enhanced models
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def load_model_safe(model_path, model_name):
    """Safely load model and tokenizer"""
    if not os.path.exists(model_path):
        print(f"❌ {model_name} model not found at {model_path}")
        return None, None
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✅ {model_name} model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=150):
    """Generate response from model"""
    if model is None or tokenizer is None:
        return "Model not available"
    
    device = next(model.parameters()).device
    
    # Format for instruction following
    formatted_prompt = f"Instruction: {prompt}\nAnswer:"
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    return generated_text

def compare_models():
    """Compare original vs enhanced models"""
    print("🇮🇳 LawGPT Model Comparison")
    print("=" * 60)
    
    # Model paths to compare
    models_to_test = [
        ("./checkpoints/lawgpt-finetuned", "Original Model"),
        ("./checkpoints/lawgpt-enhanced-finetuned", "Enhanced Model"),
        ("./checkpoints/lawgpt-enhanced-pretrained", "Enhanced Pretrained Only")
    ]
    
    # Load all available models
    loaded_models = {}
    for model_path, model_name in models_to_test:
        model, tokenizer = load_model_safe(model_path, model_name)
        if model is not None:
            loaded_models[model_name] = (model, tokenizer)
    
    if not loaded_models:
        print("❌ No models found for comparison!")
        return
    
    # Test questions covering different legal areas
    test_questions = [
        "Explain Article 21 of the Indian Constitution",
        "What is Section 302 IPC?",
        "Explain Section 9 CPC",
        "What are fundamental rights?",
        "Define murder under Indian law",
        "What is the Preamble of the Constitution?"
    ]
    
    print(f"\n🧪 Comparing {len(loaded_models)} models on {len(test_questions)} questions:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("=" * 50)
        
        for model_name, (model, tokenizer) in loaded_models.items():
            print(f"\n🤖 {model_name}:")
            print("-" * 30)
            try:
                response = generate_response(model, tokenizer, question)
                # Truncate long responses for comparison
                if len(response) > 200:
                    response = response[:200] + "..."
                print(response)
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ Model comparison completed!")
    print(f"📊 Models tested: {list(loaded_models.keys())}")

def detailed_comparison():
    """Detailed comparison with specific metrics"""
    print("🇮🇳 Detailed Model Analysis")
    print("=" * 60)
    
    # Load models
    models = {
        "Original": load_model_safe("./checkpoints/lawgpt-finetuned", "Original"),
        "Enhanced": load_model_safe("./checkpoints/lawgpt-enhanced-finetuned", "Enhanced")
    }
    
    # Filter available models
    available_models = {name: (model, tokenizer) for name, (model, tokenizer) in models.items() 
                       if model is not None}
    
    if len(available_models) < 2:
        print("❌ Need at least 2 models for detailed comparison!")
        return
    
    # Specific test cases
    test_cases = [
        {
            "category": "Constitutional Law",
            "question": "Explain the right to equality under Article 14",
            "expected_keywords": ["equality", "discrimination", "law", "state"]
        },
        {
            "category": "Criminal Law", 
            "question": "What constitutes murder under Section 302 IPC?",
            "expected_keywords": ["murder", "intention", "death", "bodily injury"]
        },
        {
            "category": "Civil Procedure",
            "question": "What is jurisdiction under CPC?",
            "expected_keywords": ["jurisdiction", "court", "civil", "procedure"]
        }
    ]
    
    print(f"📋 Testing {len(test_cases)} specific legal categories:")
    
    for test_case in test_cases:
        print(f"\n📂 Category: {test_case['category']}")
        print(f"❓ Question: {test_case['question']}")
        print("=" * 50)
        
        for model_name, (model, tokenizer) in available_models.items():
            print(f"\n🤖 {model_name} Model Response:")
            response = generate_response(model, tokenizer, test_case['question'])
            
            # Check for expected keywords
            found_keywords = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    found_keywords.append(keyword)
            
            print(response[:300] + ("..." if len(response) > 300 else ""))
            print(f"📊 Keywords found: {found_keywords} ({len(found_keywords)}/{len(test_case['expected_keywords'])})")

if __name__ == "__main__":
    print("🇮🇳 LawGPT Model Comparison Tool")
    print("Choose comparison type:")
    print("1. Quick comparison")
    print("2. Detailed analysis") 
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "2":
            detailed_comparison()
        else:
            compare_models()
    except KeyboardInterrupt:
        print("\n👋 Comparison cancelled!")
