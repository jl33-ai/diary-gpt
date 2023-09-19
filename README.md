# diary-gpt
Building ChatGPT from scratch using python, trained on my digitised diary (Large Language Model)

`Courtesy of Andrej Karpathy`
(Tutorial Used)[https://youtu.be/kCc8FmEb1nY?si=J7omX3Nlf8XTbojD]

## Results




## How it Works
- <img width="351" alt="image" src="https://github.com/jl33-ai/diary-gpt/assets/127172022/757bc8f5-afd6-42e7-a386-5709581ceee9">
- **CUDA Detection**: Checks if NVIDIA CUDA is available and identifies the number of GPUs and their specifications.
- **Hyperparameters**: Pre-defines various hyperparameters such as batch size, block size, number of iterations, learning rate, etc.
- **Data Preparation**:
  - Reads a text file (`all_thoughts_cleaned.txt`) and identifies unique characters to create an encoding and decoding mechanism.
  - Splits the text data into training (90%) and validation sets (10%).
- **Model Architecture**:
  - **Self-Attention Heads**: Uses self-attention mechanism, where each head focuses on different parts of the input.
  - **Multi-Head Attention**: Multiple attention heads operate in parallel, capturing different types of attention from the input.
  - **Feed-Forward Block**: Comprises a simple linear layer followed by a ReLU activation.
  - **Transformer Block**: Houses both the multi-head attention and feed-forward block, allowing for communication followed by computation.
  - **GPT Language Model**: Built on top of transformer blocks, this model generates text. It employs token and position embeddings and ends with a linear layer mapping to the vocabulary size.
- **Training**:
  - Utilizes the AdamW optimizer for model optimization.
  - Periodically evaluates loss on training and validation data.
  - Updates model weights based on the computed loss.
- **Generation**: After training, the model can generate new sequences of text based on the learned patterns.
- **Interactive Mode**: Users interact with the program, typing their thoughts. Depending on their choice, various operations such as writing, reading, and therapy options are executed.
