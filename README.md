# Tiny Chess Bots Challenge

Tiny Chess Bots challenge is a challenge hosted by Sebastian Lague.

An explanation of the can be found on his [YouTube](https://www.youtube.com/watch?v=iScy18pVR58).

The corresponding GitHub repository can be found [here](https://github.com/SebLague/Chess-Challenge)

We never submitted the bot even though it may have ranked fairly high in the end.

# Explanation

The bot uses a neural network written from scratch to evaluate a given chess position.
The implementation is based on matrix multiplication and addition of a bias.
The activation function of the hidden layers is `ReLu` with the activation function of the last layer bing `tanh`.

The evaluation is used in combination with classical engine algorithms, such as the negamax algorithm in combination with alpha-beta pruning. Additionally, Quiescence search is implemented.

The neural network is trained in Python using Keras with dropout layers using public evaluations of past chess games. 
To keep increase the neural network size within the token limit, the model weights are encoded as `float16` values.   
