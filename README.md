## Below are the list of NLP-Models we have in this repository with example

1. Bag-of-Words (BoW)
Concepts: Represents text as a bag of its words, disregarding grammar and word order.
Mathematical Topics:
Linear Algebra: Vector representation of text.
Probability: Word frequencies.
Example: Text classification based on word occurrences.

2. TF-IDF (Term Frequency-Inverse Document Frequency)
Concepts: Weighs words by their importance in a document relative to a corpus.
Mathematical Topics:
Statistics: Document frequencies.
Linear Algebra: Vector normalization.
Example: Information retrieval, sentiment analysis.

3. Word Embeddings (Word2Vec, GloVe)
Concepts: Dense vector representations of words capturing semantic meanings.
Mathematical Topics:
Neural Networks: Training embeddings.
Linear Algebra: Vector arithmetic.
Example: Similarity detection, text generation.

4. Named Entity Recognition (NER)
Concepts: Identifies entities (e.g., names, organizations) in text.
Mathematical Topics:
Machine Learning: Sequence labeling.
Statistics: Feature extraction.
Example: Information extraction from news articles.

5. Bidirectional Encoder Representations from Transformers (BERT)
Concepts: Transformer-based model for pre-training deep bidirectional representations.
Mathematical Topics:
Deep Learning: Transformer architecture.
Optimization: Masked language modeling.
Example: Question answering, sentiment analysis.

6. Transformer-based Models (GPT, GPT-2, GPT-3)
Concepts: Generative Pre-trained Transformers for diverse NLP tasks.
Mathematical Topics:
Attention Mechanism: Self-attention layers.
Language Modeling: Pre-training on large text corpora.
Example: Text generation, summarization.

7. Recurrent Neural Networks (RNNs) for NLP
Concept
Concept: Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining a state that evolves as it processes each element in the sequence. This ability to capture dependencies across time steps makes them suitable for tasks involving natural language, where context and order of words are crucial.
Mathematical Topics
Backpropagation Through Time (BPTT): An extension of backpropagation, adjusted for sequence data.
Matrix Operations: Linear algebra operations for hidden state transformations.
Gradient Descent: Optimization technique used to train RNNs by minimizing the loss function iteratively.
Example: Sentiment Analysis with RNN
Concept Application: RNNs are applied to sentiment analysis tasks where the goal is to classify the sentiment (positive or negative) of a text review.

Mathematical Topics Application:

Calculus: Derivatives are used in backpropagation through time (BPTT) to adjust weights.
Linear Algebra: Matrices and vectors represent hidden states and weight matrices.
Probability: Sigmoid activation function maps outputs to probabilities.
