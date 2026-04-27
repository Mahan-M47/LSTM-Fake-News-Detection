# Fake News Detection with LSTM and Hybrid CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

This repository implements a deep learning approach for fake news detection using a hybrid CNN-LSTM architecture, as proposed in the paper ["Fake news detection: A Hybrid CNN-RNN based deep learning approach"](https://www.sciencedirect.com/science/article/pii/S2667096820300070). The model is benchmarked against an LSTM-only baseline to demonstrate the benefits of combining convolutional feature extraction with recurrent sequence modeling for text classification.

## 🏗️ Model Architecture

The hybrid model follows this architecture:
1. **Embedding Layer**: Initialized with pre-trained GloVe (100d) vectors, trainable
2. **Conv1D Layer**: 128 filters, kernel size = 5, ReLU activation
3. **MaxPooling1D** Layer: pool size = 2
4. **LSTM Layer**: 32 units
5. **Dense Output Layer**: 1 unit with sigmoid activation

<br>

The LSTM-only baseline consists of:

1. **Embedding Layer**: same GloVe initialization
2. **LSTM Layer**: 32 units
3. **Dense Output Layer**: 1 unit with sigmoid activation


## 🔧  Data Preprocessing

The FA-KES (Fake and real news dataset about the middle east) dataset is used for this project. Key preprocessing steps include:

- Text data is extracted from the `article_content` column
- Maximum sequence length is set to 300 tokens
- Tokenization using TensorFlow's Tokenizer with a maximum vocabulary size of 20,000
- Sequences are padded or truncated to uniform length
- Train-test split: 80/20 ratio


### Word Embedding

GloVe (Global Vectors for Word Representation) embeddings of dimension 100 are used to initialize the embedding layer. This provides semantic word representations that capture meaningful relationships between words. The embedding layer is set as trainable to allow fine-tuning on the specific dataset.

## ⚙️ Training Details

Both models were trained with the following hyperparameters:

- Epochs: 20
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.01
- Loss Function: Binary Cross-Entropy

## 📊 Results

| Model                    | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1 (0) | F1 (1) |
|--------------------------|----------|---------------|---------------|------------|------------|--------|--------|
| Hybrid CNN + LSTM        | 0.590    | 0.54          | 0.63          | 0.54       | 0.63       | 0.54   | 0.64   |
| LSTM-only                | 0.571    | 0.54          | 0.58          | 0.28       | 0.81       | 0.37   | 0.68   |

One the validation data, the hybrid model demonstrates several advantages over the LSTM-only baseline:

- Higher overall accuracy (59.0% vs 57.1%)
- Significantly better recall for class 0 (real news): 0.54 vs 0.28
- More balanced performance across both classes
- Less bias toward predicting fake news (class 1)

The LSTM-only model shows a strong bias toward labeling articles as fake (high recall for class 1 at 0.81 but very low recall for class 0 at 0.28). The hybrid model, by extracting local features via CNN before sequence modeling, achieves a more balanced classification.

## 🚀 How to Run

1. Clone the repository
2. Download GloVe 100d embeddings (glove.6B.100d.txt) from https://nlp.stanford.edu/projects/glove/
3. Update file paths in the notebook as needed
4. Run the cells in sequence


## 📄 License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
