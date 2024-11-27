# Decoder From Scratch GPT 🐺

## Project Overview
A custom GPT (Generative Pre-trained Transformer) implementation built from scratch, following the principles outlined in the "Attention is All You Need" paper. This project demonstrates a deep dive into transformer architecture, focusing on decoder mechanisms and attention techniques.

## 🚀 Key Features
- **Model Architecture**: Custom Decoder-based Transformer
- **Parameters**: 10 Million parameters
- **Computation**: GPU (CUDA) and CPU compatible
- **Frameworks**: PyTorch, NumPy

## 📂 Project Structure
```
├── screenshot/
│   └── Screenshot.png
├── attention_is_all_you_need.pdf
├── attention_logic.ipynb
├── decoder.py
├── bigram.py
├── input.txt
├── learn.txt
├── output.txt
├── requirements.txt
└── README.md
```

## 🛠 Model Components
- Custom attention mechanism implementation
- Decoder-based text generation
- Bigram model as foundational approach
- Detailed performance tracking

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (GTX1650 Ti recommended)
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/Neupanesujal/Decoder-From-Scratch-GPT-.git
cd Decoder-From-Scratch-GPT-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# OR
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
# Execute the decoder
python decoder.py
```

## 💡 Key Learning Insights
Explore `learn.txt` for comprehensive project insights:
- Transformer architecture internals
- Attention mechanism implementations
- GPU computation optimization techniques
- Text generation strategies

## 🔍 Detailed Components
- `attention_logic.ipynb`: Core attention mechanism exploration
- `decoder.py`: Main model implementation
- `bigram.py`: Foundational language modeling approach
- `input.txt`: Training data source
- `output.txt`: Generated text results

## 📊 Performance Visualization
Check `screenshot/` for model performance metrics and loss progression visualization.

## 👤 Author
**Sujal Neupane**
- GitHub: [Neupanesujal](https://github.com/Neupanesujal)
- LinkedIn: [Sujal Neupane](https://www.linkedin.com/in/sujal-neupane/)

## 📚 References
- [Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)

## 🛡️ Troubleshooting
- Verify CUDA installation
- Check NVIDIA GPU driver compatibility
- Ensure PyTorch recognizes your GPU
