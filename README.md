# LawGPT - Indian Legal AI Assistant

Fine-tuned GPT-2 model trained on comprehensive Indian legal datasets for constitutional law, criminal law, and civil procedure queries.

## Features

- **Multi-Dataset Training**: Constitution (COI), Indian Penal Code (IPC), Code of Civil Procedure (CPC), NIA, and IndicLegalQA
- **Enhanced Knowledge**: 941 pretraining samples + 10,210 instruction samples
- **GPU Optimized**: RTX 3050 compatible with mixed precision training
- **Legal Expertise**: Constitutional law, criminal law, civil procedure, and case law

## Datasets

- **Indian Constitution (COI)**: 39 articles with detailed explanations
- **Indian Penal Code (IPC)**: 575 criminal law sections
- **Code of Civil Procedure (CPC)**: 171 civil law procedures
- **National Investigation Agency (NIA)**: 156 investigation procedures
- **IndicLegalQA**: 10,000+ legal question-answer pairs

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
git clone <repository-url>
cd LawGPT
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```
### NOTE: 
- If you have a gpu and it is not recognised then
- Download this package :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- And then test this code:
```bash
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```


### Training

#### Basic Training (Original Datasets)
```bash
python train.py
```

#### Enhanced Training (All Datasets)
```bash
python enhanced_train.py
```

#### Individual Components
```bash
python pretrain.py          # Continued pretraining only
python finetune.py          # Fine-tuning only
python enhanced_pretrain.py # Enhanced pretraining
python enhanced_finetune.py # Enhanced fine-tuning
```

### Inference

#### Interactive Chat
```bash
python inference.py
python enhanced_inference.py
```

#### Model Testing
```bash
python simple_test.py       # Quick model test
python quick_test.py        # Model loading verification
python model_comparison.py  # Compare model versions
```

## Project Structure

```
LawGPT/
├── datasets/
│   ├── COI.json                    # Indian Constitution
│   ├── ipc.json                    # Indian Penal Code
│   ├── cpc.json                    # Code of Civil Procedure
│   ├── nia.json                    # National Investigation Agency
│   └── IndicLegalQA Dataset_10K_Revised.json
├── scripts/
│   ├── data_preprocessing.py       # Basic preprocessing
│   ├── enhanced_preprocessing.py   # Multi-dataset preprocessing
│   ├── pretrain.py                # Basic pretraining
│   ├── enhanced_pretrain.py       # Enhanced pretraining
│   ├── finetune.py                # Basic fine-tuning
│   ├── enhanced_finetune.py       # Enhanced fine-tuning
│   ├── train.py                   # Complete training pipeline
│   └── enhanced_train.py          # Enhanced training pipeline
├── inference/
│   ├── inference.py               # Basic inference
│   ├── enhanced_inference.py      # Enhanced inference
│   ├── simple_test.py             # Model testing
│   ├── quick_test.py              # Quick verification
│   └── model_comparison.py        # Model comparison
├── checkpoints/                   # Model checkpoints
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Versions

### Basic Models
- `lawgpt-pretrained`: Continued pretraining on Constitution
- `lawgpt-finetuned`: Fine-tuned on constitutional Q&A

### Enhanced Models
- `lawgpt-enhanced-pretrained`: Multi-dataset pretraining (941 samples)
- `lawgpt-enhanced-finetuned`: Multi-dataset fine-tuning (10,210 samples)

## Usage Examples

### Constitutional Law
```python
question = "Explain Article 21 of the Indian Constitution"
# Response: Detailed explanation of right to life and personal liberty
```

### Criminal Law
```python
question = "What is Section 302 of the Indian Penal Code?"
# Response: Murder definition, punishment, and legal provisions
```

### Civil Procedure
```python
question = "Explain jurisdiction under CPC"
# Response: Civil court jurisdiction and procedural requirements
```

### Legal Q&A
```python
question = "What are the elements of a valid contract?"
# Response: Based on 10K+ legal Q&A training data
```

## Training Configuration

### GPU Requirements
- **Minimum**: 4GB VRAM (RTX 3050/GTX 1660)
- **Recommended**: 8GB+ VRAM (RTX 3060/4060+)
- **Batch Size**: Auto-adjusted based on GPU memory

### Training Parameters
- **Epochs**: 2-3 for enhanced training
- **Learning Rate**: 2e-5 to 5e-5
- **Mixed Precision**: Enabled for GPU efficiency
- **Gradient Accumulation**: 2-4 steps

## Performance Metrics

### Dataset Scale
- **Original**: 113 total samples
- **Enhanced**: 11,151 total samples (98x improvement)

### Training Time (RTX 3050)
- **Basic Training**: ~5-10 minutes
- **Enhanced Training**: ~30-60 minutes

### Model Capabilities
- Constitutional law explanations
- Criminal law section analysis
- Civil procedure guidance
- Legal case reasoning
- Multi-domain legal knowledge

## API Usage

### Load Model
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./checkpoints/lawgpt-enhanced-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
```

### Generate Response
```python
prompt = "Instruction: Explain Article 14\nAnswer:"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Data Sources

- **Constitution of India**: Official articles and amendments
- **Indian Penal Code**: All sections with descriptions
- **Code of Civil Procedure**: Procedural sections
- **IndicLegalQA**: Curated legal question-answer pairs
- **NIA Guidelines**: Investigation procedures

## License

This project is for educational and research purposes. Legal datasets are used under fair use for AI research.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add legal datasets or improve training
4. Submit pull request

## Disclaimer

This AI model is for educational purposes only. Always consult qualified legal professionals for actual legal advice.

## Citation

```bibtex
@misc{lawgpt2025,
  title={LawGPT: Fine-tuned GPT-2 for Indian Legal Text},
  year={2025},
  note={Multi-dataset training on Constitution, IPC, CPC, and legal Q&A}
}
```
