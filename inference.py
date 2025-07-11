"""
LawGPT Inference Script
Test the fine-tuned GPT-2 model on Indian Constitution queries
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import os

class LawGPTInference:
    def __init__(self, model_path="./checkpoints/lawgpt-finetuned"):
        """Initialize the LawGPT model for inference"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Using device: {self.device}")
        
        # Load model and tokenizer
        if os.path.exists(model_path):
            print(f"📚 Loading fine-tuned model from {model_path}...")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        else:
            print("⚠️  Fine-tuned model not found, using base GPT-2...")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def generate_response(self, prompt, max_length=300, temperature=0.7, do_sample=True):
        """Generate a response for the given prompt"""
        # Format the prompt
        formatted_prompt = f"{prompt}\nOutput:"
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.1,
                top_p=0.9,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "Output:" in response:
            response = response.split("Output:")[-1].strip()
        
        return response
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n🇮🇳 LawGPT - Indian Constitution Assistant")
        print("=" * 50)
        print("Ask me questions about the Indian Constitution!")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤖 LawGPT: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)

def test_model():
    """Test the model with sample questions"""
    lawgpt = LawGPTInference()
    
    test_questions = [
        "Explain Article 14 of the Indian Constitution",
        "What is Article 21 about?",
        "Summarize Article 1 of the Indian Constitution",
        "What does the Preamble of the Constitution say?",
        "Explain the right to equality in the Constitution"
    ]
    
    print("\n🧪 Testing LawGPT with sample questions:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 40)
        response = lawgpt.generate_response(question)
        print(f"🤖 LawGPT: {response}")
        print()

def main():
    """Main function"""
    print("🇮🇳 LawGPT Inference - Indian Constitution Assistant")
    print("Choose an option:")
    print("1. Test with sample questions")
    print("2. Interactive chat")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_model()
    elif choice == "2":
        lawgpt = LawGPTInference()
        lawgpt.interactive_chat()
    else:
        print("Invalid choice. Starting interactive chat...")
        lawgpt = LawGPTInference()
        lawgpt.interactive_chat()

if __name__ == "__main__":
    # Set seed for reproducible generation
    set_seed(42)
    main()
