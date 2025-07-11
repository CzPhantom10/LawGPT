"""
Enhanced Inference Script for Multi-Dataset Trained LawGPT
Tests knowledge from Constitution, IPC, CPC, and Legal Q&A
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def load_model(model_path):
    """Load model and tokenizer"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None, None
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=200):
    """Generate response from model"""
    device = next(model.parameters()).device
    
    # Format prompt for instruction following
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
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
        return answer
    return generated_text

def test_enhanced_model():
    """Test the enhanced model with diverse legal questions"""
    print("🇮🇳 Enhanced LawGPT Testing - Multi-Dataset Model")
    print("=" * 60)
    
    # Try to load the enhanced model first, then fallback
    model_paths = [
        "./checkpoints/lawgpt-enhanced-finetuned",
        "./checkpoints/lawgpt-finetuned",
        "./checkpoints/lawgpt-pretrained"
    ]
    
    model, tokenizer = None, None
    for model_path in model_paths:
        model, tokenizer = load_model(model_path)
        if model is not None:
            print(f"📚 Using model: {model_path}")
            break
    
    if model is None:
        print("❌ No trained model found!")
        return
    
    # Enhanced test questions covering all datasets
    test_questions = [
        # Constitution questions
        "Explain Article 21 of the Indian Constitution",
        "What does the Preamble of the Constitution say?",
        "Explain the right to equality in the Constitution",
        
        # IPC questions
        "What is Section 302 of the Indian Penal Code?",
        "Explain Section 420 IPC about cheating",
        "What does Section 498A IPC deal with?",
        
        # CPC questions
        "Explain Section 9 of the Code of Civil Procedure",
        "What is the jurisdiction under CPC?",
        
        # General legal questions
        "What is the difference between IPC and CPC?",
        "How does the Indian legal system work?",
        "What are the fundamental rights in India?",
        "Explain the concept of natural justice",
    ]
    
    print(f"🧪 Testing with {len(test_questions)} diverse legal questions:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 50)
        
        try:
            response = generate_response(model, tokenizer, question)
            print(f"🤖 LawGPT: {response}")
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
    
    print(f"\n✅ Testing completed!")

def interactive_chat():
    """Interactive chat with the enhanced model"""
    print("🇮🇳 Interactive LawGPT Chat - Enhanced Model")
    print("=" * 50)
    print("Ask questions about Indian Constitution, IPC, CPC, or legal matters.")
    print("Type 'quit' to exit.")
    print("=" * 50)
    
    # Load model
    model_paths = [
        "./checkpoints/lawgpt-enhanced-finetuned",
        "./checkpoints/lawgpt-finetuned",
        "./checkpoints/lawgpt-pretrained"
    ]
    
    model, tokenizer = None, None
    for model_path in model_paths:
        model, tokenizer = load_model(model_path)
        if model is not None:
            print(f"📚 Using model: {model_path}")
            break
    
    if model is None:
        print("❌ No trained model found!")
        return
    
    while True:
        try:
            question = input("\n🔹 Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤖 Thinking...")
            response = generate_response(model, tokenizer, question)
            print(f"🤖 LawGPT: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    print("🇮🇳 Enhanced LawGPT Inference")
    print("Choose an option:")
    print("1. Test with sample questions")
    print("2. Interactive chat")
    
    try:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            test_enhanced_model()
        elif choice == "2":
            interactive_chat()
        else:
            print("Invalid choice. Running sample test...")
            test_enhanced_model()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
