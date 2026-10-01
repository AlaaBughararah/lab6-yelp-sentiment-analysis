# Lab 6: Yelp Sentiment Analysis

This lab classifies Yelp restaurant reviews as negative (0) or positive (1) using TF-IDF, Word2Vec, and PyTorch neural networks.

## Dataset
- 1,000 reviews: 500 positive and 500 negative.
- 80% training data and 20% testing data.

## Tasks
1. Train a Word2Vec model with 200-dimensional word vectors and represent each review by averaging its word vectors.
2. Build a neural network with three hidden layers containing 128, 64, and 32 neurons, using ReLU activations.
3. Train using BCEWithLogitsLoss and Adam with a learning rate of 0.001 for 15 epochs, then calculate test accuracy.

## Results
- TF-IDF model test accuracy: **76.5%**
- Word2Vec model test accuracy: **48.5%**

The TF-IDF model achieved higher test accuracy in these runs. The models also used different architectures and training settings, so the comparison does not isolate the effect of text representation.

## Libraries
Python, Pandas, NumPy, Scikit-learn, Gensim, and PyTorch.
