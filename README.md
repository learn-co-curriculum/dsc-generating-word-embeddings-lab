
# Generating Word Embeddings - Lab

## Introduction

In this lab, you'll learn how to generate word embeddings by training a Word2Vec model, and then embedding layers into deep neural networks for NLP!

## Objectives

You will be able to:

- Train a Word2Vec model and transform words into vectors 
- Obtain most similar words by using methods associated with word vectors 


## Getting Started

In this lab, you'll start by creating your own word embeddings by making use of the Word2Vec model. Then, you'll move onto building neural networks that make use of **_Embedding Layers_** to accomplish the same end-goal, but directly in your model. 

As you've seen, the easiest way to make use of Word2Vec is to import it from the [Gensim Library](https://radimrehurek.com/gensim/). This model contains a full implementation of Word2Vec, which you can use to begin training immediately. For this lab, you'll be working with the [News Category Dataset from Kaggle](https://www.kaggle.com/rmisra/news-category-dataset/version/2#_=_).  This dataset contains headlines and article descriptions from the news, as well as categories for which type of article they belong to.

Run the cell below to import everything you'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
from gensim.models import Word2Vec
from nltk import word_tokenize
```

Now, import the data. The data is stored in the file `'News_Category_Dataset_v2.json'`.  This file is compressed, so that it can be more easily stored in a GitHub repo. **_Make sure to unzip the file before continuing!_**

In the cell below, use the `read_json()` function from Pandas to read the dataset into a DataFrame. Be sure to include the parameter `lines=True` when reading in the dataset!

Once you've imported the data, inspect the first few rows of the DataFrame to see what your data looks like. 


```python
df = None
```

## Preparing the Data

Since you're working with text data, you need to do some basic preprocessing including tokenization. Notice from the data sample that two different columns contain text data -- `headline` and `short_description`. The more text data your Word2Vec model has, the better it will perform. Therefore, you'll want to combine the two columns before tokenizing each comment and training your Word2Vec model. 

In the cell below:

* Create a column called `'combined_text'` that consists of the data from the `'headline'` column plus a space character (`' '`) plus the data from the `'short_description'` column 
* Use the `'combined_text'` column's `.map()` method and pass in `word_tokenize`. Store the result returned in `data` 


```python
df['combined_text'] = None
data = None
```

Inspect the first 5 items in `data` to see how everything looks. 


```python
data[:5]
```

Notice that although the words are tokenized, they are still in the same order they were in as headlines. This is important, because the words need to be in their original order for Word2Vec to establish the meaning of them. Remember that for a Word2Vec model you can specify a  **_Window Size_** that tells the model how many words to take into consideration at one time. 

If your window size was 5, then the model would start by looking at the words "Will Smith joins Diplo and", and then slide the window by one, so that it's looking at "Smith joins Diplo and Nicky", and so on, until it had completely processed the text example at index 1 above. By doing this for every piece of text in the entire dataset, the Word2Vec model learns excellent vector representations for each word in an **_Embedding Space_**, where the relationships between vectors capture semantic meaning (recall the vector that captures gender in the previous "king - man + woman = queen" example you saw).

Now that you've prepared the data, train your model and explore a bit!

## Training the Model

Start by instantiating a Word2Vec Model from `gensim`. 

In the cell below:

* Create a `Word2Vec` model and pass in the following arguments:
    * The dataset we'll be training on, `data`
    * The size of the word vectors to create, `size=100`
    * The window size, `window=5`
    * The minimum number of times a word needs to appear in order to be counted in  the model, `min_count=1` 
    * The number of threads to use during training, `workers=4`


```python
model = None
```

Now, that you've instantiated Word2Vec model, train it on your text data. 

In the cell below:

* Call the `.train()` method on your model and pass in the following parameters:
    * The dataset we'll be training on, `data`
    * The `total_examples`  of sentences in the dataset, which you can find in `model.corpus_count` 
    * The number of `epochs` you want to train for, which we'll set to `10`


```python

```

Great! You now have a fully trained model! The word vectors themselves are stored in the `Word2VecKeyedVectors` instance, which is stored in the `.wv` attribute. To simplify this, restore this object inside of the variable `wv` to save yourself some keystrokes down the line. 


```python
wv = None
```

## Examining Your Word Vectors

Now that you have a trained Word2Vec model, go ahead and explore the relationships between some of the words in the corpus! 

One cool thing you can use Word2Vec for is to get the most similar words to a given word. You can do this by passing in the word to `wv.most_similar()`. 

In the cell below, try getting the most similar word to `'Texas'`.


```python

```

Interesting! All of the most similar words are also states. 

You can also get the least similar vectors to a given word by passing in the word to the `.most_similar()` method's `negative` parameter. 

In the cell below, get the least similar words to `'Texas'`.


```python

```

This seems like random noise. It is a result of the way Word2Vec is computing the similarity between word vectors in the embedding space. Although the word vectors closest to a given word vector are almost certainly going to have similar meaning or connotation with your given word, the word vectors that the model considers 'least similar' are just the word vectors that are farthest away, or have the lowest cosine similarity. It's important to understand that while the closest vectors in the embedding space will almost certainly share some level of semantic meaning with a given word, there is no guarantee that this relationship will hold at large distances. 

You can also get the vector for a given word by passing in the word as if you were passing in a key to a dictionary. 

In the cell below, get the word vector for `'Texas'`.


```python

```

Now get all of the word vectors from the object at once. You can find these inside of `wv.vectors`. Try it out in the cell below.  


```python

```

As a final exercise, try to recreate the _'king' - 'man' + 'woman' = 'queen'_ example previously mentioned. You can do this by using the `.most_similar()` method and translating the word analogies into an addition/subtraction formulation (as shown above). Pass the original comparison, which you are calculating a difference between, to the negative parameter, and the analogous starter you want to apply the same transformation to, to the `positive` parameter.

Do this now in the cell below. 


```python

```

As you can see from the output above, your model isn't perfect, but 'Queen' and 'Princess' are still in the top 5. As you can see from the other word in top 5, 'reminiscent' -- your model is far from perfect. This is likely because you didn't have enough training data. That said, given the small amount of training data provided, the model still performs remarkably well! 

In the next lab, you'll reinvestigate transfer learning, loading in the weights from an open-sourced model that has already been trained for a very long time on a massive amount of data. Specifically, you'll work with the GloVe model from the Stanford NLP Group. There's not really any benefit from training the model ourselves, unless your text uses different, specialized vocabulary that isn't likely to be well represented inside an open-source model.

## Summary

In this lab, you learned how to train and use a Word2Vec model to create vectorized word embeddings!
