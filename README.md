# Gender_Classification
# 🚀 MobileNetV2 Feature Extractor with Cross-Entropy Loss

📂 This repository contains an implementation of a MobileNetV2-based feature extractor model for image classification, trained with Cross-Entropy Loss and optimized using Adam optimizer.

## 🧠 Model Overview

- **🏗️ Architecture**: MobileNetV2 (pre-trained)
- **🔧 Feature Extraction**: Fine-tuned for custom tasks
- **📉 Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss()`)
- **⚙️ Optimizer**: Adam (`torch.optim.Adam`)
- **🖼️ Input Size**: 224×224 pixels (3 channels - RGB)

## 🏋️ Training Details

| Parameter          | Value              |
|--------------------|--------------------|
| 📚 Learning Rate   | 0.001              |
| 🧩 Batch Size      | 32                 |
| ⏳ Epochs          | 5 |
| 💾 Dataset         | FACECOM|


## Results on validation set
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Task_A | 92.18% | 95.06% | 95.34% | 95.30% |

## � Deployment

🌍 **Hugging Face Model Hub**:  
🔗 [GenderAI🤖 ♀️♂️](https://huggingface.co/spaces/rxhxss/Gender_Classification)

