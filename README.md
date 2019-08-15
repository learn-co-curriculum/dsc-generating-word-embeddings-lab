
# Generating Word Embeddings - Lab

## Introduction

In this lab, you'll learn how to generate word embeddings by training a Word2Vec model, and then embedding layers into Deep Neural Networks for NLP!

## Objectives

You will be able to:

* Demonstrate a basic understanding of the architecture of the Word2Vec model
* Demonstrate an understanding of the various tunable parameters of Word2Vec such as vector size and window size

## Getting Started

In this lab, you'll start by creating your own word embeddings by making use of the Word2Vec Model. Then, you'll move onto building Neural Networks that make use of **_Embedding Layers_** to accomplish the same end-goal, but directly in your model. 

As you've seen, the easiest way to make use of Word2Vec is to import it from the [Gensim Library](https://radimrehurek.com/gensim/). This model contains a full implementation of Word2Vec, which you can use to begin training immediately. For this lab, you'll be working with the [News Category Dataset from Kaggle](https://www.kaggle.com/rmisra/news-category-dataset/version/2#_=_).  This dataset contains headlines and article descriptions from the news, as well as categories for which type of article they belong to.

Run the cell below to import everything you'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
from gensim.models import Word2Vec
from nltk import word_tokenize
```

Now, import the data. The data stored in the file `'News_Category_Dataset_v2.json'`.  This file is compressed, so that it can be more easily stored in a github repo. **_Make sure to unzip the file before continuing!_**

In the cell below, use the `read_json` function from pandas to read the dataset into a DataFrame. Be sure to include the parameter `lines=True` when reading in the dataset!

Once you've loaded in the data, inspect the head of the DataFrame to see what your data looks like. 


```python
raw_df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>category</th>
      <th>date</th>
      <th>headline</th>
      <th>link</th>
      <th>short_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Melissa Jeltsen</td>
      <td>CRIME</td>
      <td>2018-05-26</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Andy McDonald</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Jim Carrey Blasts 'Castrato' Adam Schiff And D...</td>
      <td>https://www.huffingtonpost.com/entry/jim-carre...</td>
      <td>The actor gives Dems an ass-kicking for not fi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Julianna Margulies Uses Donald Trump Poop Bags...</td>
      <td>https://www.huffingtonpost.com/entry/julianna-...</td>
      <td>The "Dietland" actress said using the bags is ...</td>
    </tr>
  </tbody>
</table>
</div>



## Preparing the Data

Since you're working with text data, you need to do some basic preprocessing including tokenization. Notice from the data sample that two different columns contain text data--`headline` and `short_description`. The more text data your Word2Vec model has, the better it will perform. Therefore, you'll want to combine the two columns before tokenizing each comment and training your Word2Vec model. 

In the cell below:

* Create a column called `combined_text` that consists of the data from `df.headline` plus a space character (`' '`) plus the data from `df.short_description`.
* Use the `combined_text` column's `map()` function and pass in `word_tokenize`. Store the result returned in `data`.


```python
df['combined_text'] = df.headline + ' ' +  df.short_description
data = df['combined_text'].map(word_tokenize)
```

Inspect the first 5 items in `data` to see how everything looks. 


```python
data[:5]
```




    0    [There, Were, 2, Mass, Shootings, In, Texas, L...
    1    [Will, Smith, Joins, Diplo, And, Nicky, Jam, F...
    2    [Hugh, Grant, Marries, For, The, First, Time, ...
    3    [Jim, Carrey, Blasts, 'Castrato, ', Adam, Schi...
    4    [Julianna, Margulies, Uses, Donald, Trump, Poo...
    Name: combined_text, dtype: object



Notice that although the words are tokenized, they are still in the same order they were in as headlines. This is important, because the words need to be in their original order for Word2Vec to establish the meaning of them. Remember that for a Word2Vec model you can specify a  **_Window Size_** that tells the model how many words to take into consideration at one time. 

If your window size was 5, then the model would start by looking at the words "Will Smith joins Diplo and", and then slide the window by one, so that it's looking at "Smith joins Diplo and Nicky", and so on, until it had completely processed the text example at index 1 above. By doing this for every piece of text in the entire dataset, the Word2Vec model learns excellent vector representations for each word in an **_Embedding Space_**, where the relationships between vectors capture semantic meaning (recall the vector that captures gender in the previous "king - man + woman = queen" example you saw).

Now that you've prepared the data, train your model and explore a bit!

## Training the Model

Start by instantiating a Word2Vec Model from gensim below. 

In the cell below:

* Create a `Word2Vec` model and pass in the following arguments:
    * The dataset we'll be training on, `data`
    * The size of the word vectors to create, `size=100`
    * The window size, `window=5`
    * The minimum number of times a word needs to appear in order to be counted in  the model, `min_count=1`.
    * The number of threads to use during training, `workers=4`


```python
model = Word2Vec(data, size=100, window=5, min_count=1, workers=4)
```

Now, that you've instantiated Word2Vec model, train it on your text data. 

In the cell below:

* Call `model.train()` and pass in the following parameters:
    * The dataset we'll be training on, `data`
    * The `total_examples`  of sentences in the dataset, which you can find in `model.corpus_count`. 
    * The number of `epochs` you want to train for, which we'll set to `10`


```python
model.train(data, total_examples=model.corpus_count, epochs=10)
```




    (55562550, 67339860)



Great! you now have a fully trained model! The word vectors themselves are stored inside of a `Word2VecKeyedVectors` instance, which is stored inside of `model.wv`. To simplify this, restore this object inside of the variable `wv` to save yourself some keystrokes down the line. 


```python
wv = model.wv
```

## Examining Your Word Vectors

Now that you have a trained Word2Vec model, go ahead and explore the relationships between some of the words in the corpus! 

One cool thing you can use Word2Vec for is to get the most similar words to a given word. You can do this passing in the word to `wv.most_similar()`. 

In the cell below, try getting the most similar word to `'Texas'`.


```python
wv.most_similar('Texas')
```




    [('Ohio', 0.8216778039932251),
     ('Maryland', 0.820242702960968),
     ('Pennsylvania', 0.8115444183349609),
     ('Oklahoma', 0.8059725761413574),
     ('Georgia', 0.7871538400650024),
     ('Louisiana', 0.7868552803993225),
     ('Oregon', 0.7828892469406128),
     ('Connecticut', 0.7785530686378479),
     ('Wisconsin', 0.7663697600364685),
     ('Utah', 0.7626876831054688)]



Interesting! All of the most similar words are also states. 

You can also get the least similar vectors to a given word by passing in the word to the `most_similar()` function's `negative` parameter. 

In the cell below, get the least similar words to `'Texas'`.


```python
wv.most_similar(negative='Texas')
```




    [('much-vaunted', 0.4367696940898895),
     ('Parent/Grandparent', 0.4149670898914337),
     ('once-reliable', 0.41059452295303345),
     ('Unelectable', 0.3959532678127289),
     ('Opprtunism', 0.39321935176849365),
     ('Double-parked', 0.3905583620071411),
     ('Sergeant-at-Arms', 0.37530481815338135),
     ('Likened', 0.3672611117362976),
     ('maitre', 0.36494070291519165),
     ('Un-Blind', 0.3625698685646057)]



This seems like random noise. It is a result of the way Word2Vec is computing the similarity between word vectors in the embedding space. Although the word vectors closest to a given word vector are almost certainly going to have similar meaning or connotation with your given word, the word vectors that the model considers 'least similar' are just the word vectors that are farthest away, or have the lowest cosine similarity. It's important to understand that while the closest vectors in the embedding space will almost certainly share some level of semantic meaning with a given word, there is no guarantee that this relationship will hold at large distances. 

You can also get the vector for a given word by passing in the word as if you were passing in a key to a dictionary. 

In the cell below, get the word vector for `'Texas'`.


```python
wv['Texas']
```




    array([-1.2401885 ,  0.07316723,  0.5426485 , -1.9113084 , -0.9974326 ,
            1.5531368 ,  0.8876857 ,  0.9750643 ,  0.92900705,  1.523192  ,
            2.3522954 ,  1.0757657 , -0.74888057,  2.2034118 , -0.26056725,
            0.14731078,  0.7483212 , -2.2312248 ,  0.2914787 ,  3.28357   ,
            0.49191797, -0.5155347 ,  0.81373286,  1.3505329 , -0.02592773,
            0.60989344, -3.5890887 ,  1.9001029 , -1.7027069 , -1.5107026 ,
           -1.19016   ,  0.2772145 , -0.70366204, -0.85404295,  1.3004638 ,
           -1.856721  ,  0.8884504 ,  0.01409801,  0.37409958, -1.3967689 ,
           -1.8351244 , -1.7225714 , -0.24079297,  1.4265858 , -0.43965214,
           -0.33554283, -0.24751939,  0.14829023,  0.6424359 ,  0.7000424 ,
            1.8553067 ,  0.6845079 , -0.11181285,  1.2791896 ,  2.5373924 ,
            0.18910365, -1.2802954 ,  1.4823729 ,  0.97811407, -0.40233034,
           -0.5029891 ,  3.0834503 ,  2.1235855 ,  0.095056  , -0.05742403,
            1.1933752 , -0.4946205 , -0.17487265,  0.6604149 ,  2.519939  ,
           -0.26483992,  1.655191  ,  1.8373055 ,  0.76514614, -1.7379202 ,
           -0.9396014 , -3.7663083 , -2.162103  ,  0.05214275, -1.6772201 ,
            2.0532708 ,  1.5082473 , -0.7698685 , -1.1882211 , -1.3108006 ,
            0.26351514, -0.71885866,  1.0104517 ,  0.02348135, -0.28737685,
           -0.32011354,  0.31973317, -0.9617833 , -1.0241627 , -0.37533018,
           -1.1554545 , -0.15405892, -0.840807  , -0.9584556 , -1.4261397 ],
          dtype=float32)



Now get all of the word vectors from the object at once. You can find these inside of `wv.vectors`. Try it out in the cell below.  


```python
wv.vectors
```




    array([[-1.9846559e-01, -5.5181438e-01, -7.6076186e-01, ...,
             9.5231330e-01, -8.8179910e-01,  1.6026921e+00],
           [-1.0904990e+00, -1.2622950e+00,  4.4958344e-01, ...,
            -5.0044709e-01, -1.5242363e+00,  5.2104121e-01],
           [-2.2713695e+00, -2.0855942e+00,  9.3830399e-02, ...,
             6.8697584e-01, -1.1908143e+00,  4.2356486e+00],
           ...,
           [ 3.2912325e-02, -5.4993749e-02,  3.3443581e-02, ...,
             1.2652363e-01, -1.5986431e-02,  1.4026050e-02],
           [ 1.2474794e-02, -3.4032460e-02,  3.2994132e-02, ...,
            -2.8819051e-02, -7.1466140e-02, -2.1249249e-03],
           [ 5.6088651e-03, -3.1472065e-02, -3.8587283e-02, ...,
             2.1852210e-02,  1.9721478e-02,  4.3979585e-02]], dtype=float32)



As a final exercise, try to recreate the _'king' - 'man' + 'woman' = 'queen'_ example previously mentioned. You can do this by using the `most_similar` function and translating the word analogies into an addition/subtraction formulation (as shown above). Pass the original comparison, which you are calculating a difference between, to the negative parameter, and the analogous starter you want to apply the same transformation to, to the `positive` parameter.

Do this now in the cell below. 


```python
wv.most_similar(positive=['king', 'woman'], negative=['man'])
```




    [('reminiscent', 0.6167434453964233),
     ('crown', 0.607818603515625),
     ('queen', 0.5999212861061096),
     ('title', 0.5968424081802368),
     ('princess', 0.5878039002418518),
     ('revival', 0.5792500972747803),
     ('birthplace', 0.5727936625480652),
     ('diva', 0.5674294233322144),
     ('symbol', 0.5663068294525146),
     ('goddess', 0.5661177039146423)]



As you can see from the output above, your model isn't perfect, but 'Queen' is still in the top 3, and with 'Princess' not too far behind. As you can see from the word in first place, 'reminiscent', your model is far from perfect. This is likely because you didn't have enough training data. That said, given the small amount of training data provided, the model still performs remarkably well! 

In the next lab, you'll reinvestigate transfer learning, loading in the weights from an open-sourced model that has already been trained for a very long time on a massive amount of data. Specifically, you'll work with the GloVe model from the Stanford NLP Group. There's not really any benefit from training the model ourselves, unless your text uses different, specialized vocabulary that isn't likely to be well represented inside an open-source model.

## Summary

In this lab, you learned how to train and use a Word2Vec model to created vectorized word embeddings!
