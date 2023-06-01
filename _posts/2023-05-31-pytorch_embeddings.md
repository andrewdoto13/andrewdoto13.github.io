---
title: "Exploring Embeddings in PyTorch"
classes: wide
tags: [python, pytorch, nlp, embeddings, deep-learning]
desc: "Working with non-numeric data in PyTorch"
---

Working with text data or natural language data is very common in data science and machine learning, and this is broadly considered its own sub-field commonly called natural language processing (NLP). As of March 2023, recent developments in the field of NLP, particularly the application of state-of-the-art deep learning techniques, have taken the world by storm. ChatGPT, a chatbot-like service from the company OpenAI, has garnered widespread attention and has put generative AI into the zeitgeist in a major way.

Stepping back from the hype of the most recent advancements, let's take a look at a simple concept and answer a fundamental beginning question. Consider this: broadly, machine learning and AI operate primarily on numerical representations of data, especially neural networks. With that said, how can the models operate on text data; after all, you can't perform math operations on words and sentences. There must be some way to represent natural language in some numerical way. Let's consider some ways to answer this question.

## One-hot vectors

One common way of representing words is in a type of vector called a one-hot vector. A one-hot vector is a vector of all zeros, except for one location with a one, and that location is unique to that word. Let's take a look at a trivial example below to illustrate.

In the example, it takes a "vocabulary" or "corpus" that is comprised of only 3 words, "I like dogs." The dictionary contains the one-hot vector representations of each of the three words. You'll notice how the length of each vector is 3, the size of the total "vocabulary."


```python
import torch
import torch.nn as nn

word_vectors = {
    "I": torch.tensor([1,0,0]),
    "like": torch.tensor([0,1,0]),
    "dogs": torch.tensor([0,0,1])
}

print(word_vectors["dogs"])
```

    tensor([0, 0, 1])


One-hot vectors are a simple way to represent text data, and they are still used frequently in traditional machine learning. However, they are not without flaws. First, a somewhat obvious flaw is the dependence of the vector length on the size of the vocabulary. In applications where the corpus is extensive, this makes one-hot vectors an intractible or inefficient solution. 

Another flaw is that one-hot vectors can't contain semantic similarity between words. Notice in the example above, each vector is similarly different from all of the other vectors. It would be ideal if vectors could be created such that a similarity score could be calculated between vectors and therefore "semantic information" can be encoded in them. Enter embeddings, which we will explore below in the PyTorch library.

## PyTorch Embeddings

In the example below, we will use the same trivial vocabulary example. In order to translate our words into dense vectors (vectors that are not mostly zero), we can use the Embedding class provided by PyTorch. We initialize the embedding by passing in the number of words in our vocab and then the desired size of the vectors produced by the embedding. The embedding that is produced is exactly what we are looking for, but of course, the numbers themselves don't have any meaning to us if we inspect it.


```python
word_ix = {
    "I": 0,
    "like": 1,
    "dogs": 2
}

# vocab length of 3, embedding size of 5
embeddings = nn.Embedding(3, 5)
dogs_tensor = torch.tensor(word_ix["dogs"])

dogs_embedding = embeddings(dogs_tensor)
print(dogs_embedding)
```

    tensor([-1.2571,  0.7301, -0.1999,  1.5184,  0.6683],
           grad_fn=<EmbeddingBackward0>)


The wonderful thing about using PyTorch embeddings is that the embeddings are actually trainable. So during training of a deep neural network for example, backpropagation can help this embedding layer learn these representations as part of the overall optimization, and you can think of it as a kind of trainable lookup table that stores relationships between words. This is just scratching the surface, but I hope you enjoyed this brief introduction to PyTorch embeddings.
