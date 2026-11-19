# diploma-master

A master's thesis repository containing theoretical materials, reference literature, and source code on artificial neural networks, deep learning, and medical image analysis.

## Repository Contents

| Folder or File | Description |
| --- | --- |
| [Code](./Code/) | PyTorch implementation of the EfficientNet family, including training and evaluation code. |
| [Literature](./Literature/) | PDF reference materials on neural networks, deep learning, and artificial intelligence in medicine. |
| [Part_1.pdf](./Part_1.pdf) | Armenian-language introduction to neural network architecture, with handwritten digit recognition and perceptron examples. |
| [Mathematical Foundations of Deep Learning (DOCX)](./Խորը_ուսուցման_մաթեմատիկական_հիմունքներ.docx) | Armenian-language theoretical material covering neural networks, CNNs, ViTs, and the mathematical foundations of four model architectures. |
| [README.md](./README.md) | Project overview and guide to the repository's contents. |

## Mathematical Foundations of Deep Learning

[Open the Word document](./Խորը_ուսուցման_մաթեմատիկական_հիմունքներ.docx)

This Armenian-language document presents the mathematical foundations of artificial neural networks, convolutional neural networks (CNNs), and Vision Transformers (ViTs). It consists of two main chapters.

### Chapter 1: Theoretical and Mathematical Foundations of Artificial Neural Networks

- Artificial intelligence, machine learning, and deep learning.
- Artificial neurons, perceptrons, and multilayer perceptrons (MLPs).
- Activation functions and loss functions.
- Gradient-based optimization, Momentum, and Adam.
- Backpropagation, generalization, overfitting, and regularization.

### Chapter 2: Mathematical Foundations of CNNs and ViTs and the Architectures Studied

- Convolution, feature maps, padding, stride, dilation, pooling, and normalization.
- CNN training, computational complexity, receptive fields, and Grad-CAM.
- Image patching, positional embeddings, and the classification token.
- Self-attention, multi-head attention, and the Transformer encoder.
- Architecture and comparison of ResNet50, DenseNet121, EfficientNet-B7, and ViT-B/16.
- Transfer learning and applications to dermoscopic image classification.

The document includes mathematical formulas, illustrations, comparison tables, and references.

## Neural Network Architecture

[Part_1.pdf](./Part_1.pdf) introduces neural network fundamentals through handwritten digit recognition, perceptrons, and their organization into layers.

## Reference Literature

The [Literature](./Literature/) folder contains the following PDF materials:

| Resource | Topic |
| --- | --- |
| [AI in Dermatology](./Literature/AI_in_Dermatology.pdf) | Artificial intelligence in dermatology. |
| [Artificial Intelligence in Oncology](./Literature/Artificial_Intelligence_in_Oncology.pdf) | Artificial intelligence in oncology. |
| [Deep Learning in Medical Imaging](./Literature/Deep%20Learning%20in%20Medical%20Imaging.pdf) | Deep learning for medical image analysis. |
| [Neural Networks and Deep Learning](./Literature/Neural%20Networks%20and%20Deep%20Learning.pdf) | Fundamentals of neural networks and deep learning. |

## Source Code

The [Code](./Code/) folder contains a PyTorch implementation of the **EfficientNet B0-B7** family. The current [main.py](./Code/main.py) selects **EfficientNet-B1** as the training model.

| File or Folder | Purpose |
| --- | --- |
| [Code/README.md](./Code/README.md) | Implementation overview and instructions for dataset organization, training, and evaluation. |
| [Code/main.py](./Code/main.py) | Data loading, model training and evaluation, and checkpoint saving. |
| [Code/main.sh](./Code/main.sh) | Command for launching distributed training. |
| [Code/requirements.txt](./Code/requirements.txt) | Python dependencies. |
| [Code/efficientnet/models](./Code/efficientnet/models/) | EfficientNet model architectures. |
| [Code/efficientnet/loss](./Code/efficientnet/loss/) | Loss function implementations. |
| [Code/efficientnet/optim](./Code/efficientnet/optim/) | Optimization components, including RMSprop. |
| [Code/efficientnet/scheduler](./Code/efficientnet/scheduler/) | Learning rate scheduling. |
| [Code/efficientnet/utils](./Code/efficientnet/utils/) | Data processing, augmentation, evaluation metrics, and distributed computing utilities. |
| [Code/weights/last.pth](./Code/weights/last.pth) | Saved model weights. |
| [Code/assets/cat.jpg](./Code/assets/cat.jpg) | Example image. |

The code expects `train` and `val` directories, each containing a separate subdirectory for every class. The dataset location is configured through the `--data-path` argument. The current default path points to ImageNet; the dataset must be provided separately.

See the [code README](./Code/README.md) for execution details.

## Suggested Reading Order

1. Start with [Part_1.pdf](./Part_1.pdf) for an introduction to neural networks.
2. Read the [Word document](./Խորը_ուսուցման_մաթեմատիկական_հիմունքներ.docx) for mathematical formulations and CNN/ViT architectures.
3. Explore the [Literature](./Literature/) folder for additional references.
4. Review the [Code](./Code/) folder to study the EfficientNet implementation.
