"""
LawGPT Training Pipeline
Pretrain + Fine-tune GPT-2 on Indian Constitution Data
"""
import torch
import os
from pretrain import pretrain_gpt2
from finetune import finetune_gpt2

def main():
    print("🇮🇳 LawGPT Training Pipeline - Indian Constitution")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slow)")
    
    print("\n🎯 Training Steps:")
    print("1. Continued Pretraining on Constitution Articles")
    print("2. Supervised Fine-tuning on Q&A Instructions")
    print("=" * 60)
    
    # Step 1: Pretraining
    print("\n📖 STEP 1: PRETRAINING")
    print("-" * 30)
    try:
        pretrained_model_path = pretrain_gpt2()
        print(f"✅ Pretraining completed successfully!")
        
        # Clear GPU cache between training phases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
            
    except Exception as e:
        print(f"❌ Pretraining failed: {e}")
        print("🔄 Continuing with base GPT-2 model...")
        pretrained_model_path = None
    
    # Step 2: Fine-tuning
    print("\n🎯 STEP 2: FINE-TUNING")
    print("-" * 30)
    try:
        finetuned_model_path = finetune_gpt2(pretrained_model_path)
        print(f"✅ Fine-tuning completed successfully!")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return
    
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print("📁 Model Locations:")
    if pretrained_model_path:
        print(f"📚 Pretrained: {pretrained_model_path}")
    print(f"🎯 Fine-tuned: {finetuned_model_path}")
    print("\n🧪 Next Steps:")
    print("- Run inference.py to test the model")
    print("- Ask questions about Indian Constitution articles")

if __name__ == "__main__":
    main()
