"""
Enhanced Training Pipeline for Multiple Legal Datasets
Handles: Constitution, IPC, CPC, NIA, and IndicLegalQA (10K+ samples)
"""
import torch
import os
from enhanced_pretrain import enhanced_pretrain_gpt2
from enhanced_finetune import enhanced_finetune_gpt2

def main():
    """Enhanced training pipeline with multiple legal datasets"""
    print("🇮🇳 Enhanced LawGPT Training Pipeline - Multiple Legal Datasets")
    print("=" * 70)
    print("📚 Datasets included:")
    print("   • Indian Constitution (COI)")
    print("   • Indian Penal Code (IPC)")
    print("   • Code of Civil Procedure (CPC)")
    print("   • National Investigation Agency (NIA)")
    print("   • IndicLegalQA Dataset (10K+ Q&A pairs)")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("⚠️  No GPU available, using CPU (training will be slow)")
    
    print("\n🎯 Training Pipeline:")
    print("1. Enhanced Continued Pretraining (941 samples)")
    print("2. Enhanced Supervised Fine-tuning (10,210 samples)")
    print("=" * 70)
    
    try:
        # Step 1: Enhanced Pretraining
        print("\n📖 STEP 1: ENHANCED PRETRAINING")
        print("-" * 40)
        enhanced_pretrained_model_path = enhanced_pretrain_gpt2()
        print(f"✅ Enhanced pretraining completed successfully!")
        
        # Clear GPU cache between training phases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
        
        # Step 2: Enhanced Fine-tuning
        print(f"\n🎯 STEP 2: ENHANCED FINE-TUNING")
        print("-" * 40)
        enhanced_finetuned_model_path = enhanced_finetune_gpt2(enhanced_pretrained_model_path)
        print(f"✅ Enhanced fine-tuning completed successfully!")
        
        # Final success message
        print("\n🎉 ENHANCED TRAINING COMPLETED!")
        print("=" * 70)
        print("📁 Model Locations:")
        print(f"   📚 Enhanced Pretrained: {enhanced_pretrained_model_path}")
        print(f"   🎯 Enhanced Fine-tuned: {enhanced_finetuned_model_path}")
        print("\n📊 Dataset Summary:")
        print("   🔄 Pretraining: 941 samples (25x increase)")
        print("   🎯 Instruction: 10,210 samples (138x increase)")
        print("\n💡 Next Steps:")
        print("   • Test with: python enhanced_inference.py")
        print("   • Compare with: python model_comparison.py")
        print("   • The model now understands:")
        print("     - Indian Constitution articles")
        print("     - Indian Penal Code sections")
        print("     - Civil Procedure Code")
        print("     - Legal case Q&A (10K+ examples)")
        
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
