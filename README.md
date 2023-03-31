# volgpt

### Code repo for post: Predicting volatility with NanoGPT

This work explores use of text-to-text LLMs for vol prediction, something normally done with number-to-number stochastic volatility model such as the MSM or Heston, with high frequency data.

Implements NanoGPT using PyTorch in the single python file nanogpt.py. The function train_and_generate takes in a text file path, trains the model on the text in the file, and then generates new text, in this case high-frequency OHLC data. 

The main components of the code are:

> 1. Data preparation: Reads the input text file and encodes the characters into integer tokens. The data is then split into train, validation, and test sets, the latter to enable performance evaluation (MSE/MAE).
> 2. Data loading: Function get_batch is defined to create batches of input and target sequences.
> 3. Model components: Define several custom classes to implement the Transformer architecture, including Head, MultiHeadAttention, FeedFoward, Block, and BigramLanguageModel.
> 4. Model instantiation: Creates an instance of the BigramLanguageModel and moves it to the GPU (if available, I am using a NVIDIA GeForce RTX 3080 Ti Laptop GPU).
> 5. Training loop: Train the model for a specified number of iterations with an AdamW optimizer, periodically evaluating the train and validation losses.
> 6. Text generation: After training, the model generates new text using the generate method of the BigramLanguageModel class.

Provide the path to a text file as the text_file_path argument to the train_and_generate function and adjust the other function parameters like max_iters, learning_rate, and max_new_tokens as needed. The function will return the generated text.
