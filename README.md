# PPGPT

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A comprehensive system for predicting physiological parameters from PPG signals using deep learning*

[Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Paper](#citation)

</div>

---

## 🌟 Features

- **🔀 Multi-Modal Learning**: Dual-stream encoder for signal and spatial features
- **👥 Multi-Expert Architecture**: Specialized networks for different physiological parameters
- **🤖 Transformer Models**: State-of-the-art attention mechanisms for PPG analysis
- **📊 Comprehensive Evaluation**: Built-in metrics and visualization tools
- **🚀 Easy Deployment**: Docker support and streamlined workflows
- **⚡ High Performance**: Optimized for both accuracy and efficiency

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Example
```bash
python scripts/preprocess_data.py --input_path sample_data.csv
python scripts/train.py --config configs/dual_stream_config.yaml
python scripts/evaluate.py --config configs/dual_stream_config.yaml --model_path outputs/final_model.pth
```

### 3. Full Experiments
```bash
# Run all experiments with evaluation
python scripts/run_experiments.py --experiments all --evaluate
```

## 📁 Project Structure

```
ppg-physiological-prediction/
├── 📦 config/                 # Configuration files
│   ├── model_config.py        # Model configurations
│   └── expert_config.py       # Expert network configs
├── 📊 data/                   # Data processing
│   ├── dataset.py             # Dataset classes
│   ├── preprocessing.py       # Signal preprocessing
│   └── data_generator.py      # Data generation pipeline
├── 🧠 models/                 # Model architectures
│   ├── dual_stream_encoder.py # Multi-modal encoder
│   ├── expert_model.py        # Multi-expert networks
│   ├── transformer_model.py   # Transformer models
│   └── base_model.py          # Base model class
├── 🎯 training/               # Training utilities
│   ├── trainer.py             # Training loops
│   └── losses.py              # Loss functions
├── 🔧 utils/                  # Utilities
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Plotting functions
│   └── feature_extraction.py  # Feature extraction
├── 📜 scripts/                # Execution scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── run_experiments.py     # Experiment runner
├── ⚙️ configs/                # YAML configurations
├── 🧪 tests/                  # Unit tests
└── 📖 docs/                   # Documentation
```

## 🏗️ Model Architectures

### 🔀 Dual-Stream Encoder
Processes both 1D PPG signals and 2D spatial representations:
```python
from models.dual_stream_encoder import DualStreamEncoder

model = DualStreamEncoder(
    signal_dim=600,
    image_channels=3,
    output_dim=256
)
```

### 👥 Multi-Expert Model
Specialized experts for different physiological parameters:
```python
from models.expert_model import MultiExpertModel

model = MultiExpertModel(
    input_dim=256,
    expert_names=['SBP', 'DBP', 'HR', 'GLU']
)
```

### 🤖 Transformer Model
Attention-based sequence modeling:
```python
from models.transformer_model import PPGTransformer

model = PPGTransformer(
    input_dim=600,
    d_model=256,
    num_heads=8,
    num_layers=6
)
```

## ⚙️ Configuration

### Model Configuration
```yaml
# configs/dual_stream_config.yaml
model:
  type: dual_stream
  config:
    signal_dim: 600
    image_channels: 3
    output_dim: 256

training:
  batch_size: 64
  epochs: 100
  optimizer:
    type: adamw
    lr: 1e-3
```

### Custom Configuration
```python
from config.model_config import ModelConfig

config = ModelConfig(
    signal_dim=600,
    hidden_dim=256,
    num_heads=8
)
```

### Data Preprocessing
```bash
python scripts/preprocess_data.py \
    --input_path raw_data.csv \
    --output_dir processed_data \
    --signal_length 600 \
    --quality_check
```

## 🎯 Training & Evaluation

### Training
```bash
# Single model training
python scripts/train.py --config configs/dual_stream_config.yaml
```

### Evaluation
```bash
# Evaluate with visualization
python scripts/evaluate.py \
    --config configs/dual_stream_config.yaml \
    --model_path model.pth \
    --visualize
```

### Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Pearson**: Pearson Correlation Coefficient

## 🧪 Testing

```bash
# Run all tests
make test

# With coverage
make test-coverage

# Specific test
python -m pytest tests/test_models.py -v
```

## 📚 API Reference

### Core Classes
- `DualStreamEncoder`: Multi-modal signal processing
- `MultiExpertModel`: Multi-task prediction
- `PPGTransformer`: Attention-based modeling
- `Trainer`: Training orchestration
- `MetricsTracker`: Performance monitoring

### Key Functions
- `preprocess_ppg_signal()`: Signal preprocessing
- `generate_spatial_features()`: Spatial feature extraction  
- `calculate_metrics()`: Evaluation metrics
- `plot_predictions_vs_actual()`: Visualization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={}
}
```

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Contributors to the PPG signal processing community
- Research institutions supporting this work

---

<div align="center">

**[⭐ Star this repo]()** if it helped you!

Made with ❤️ by the PPG Research Team

</div>
