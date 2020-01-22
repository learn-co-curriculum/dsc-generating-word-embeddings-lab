
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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.head()
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

Since you're working with text data, you need to do some basic preprocessing including tokenization. Notice from the data sample that two different columns contain text data -- `headline` and `short_description`. The more text data your Word2Vec model has, the better it will perform. Therefore, you'll want to combine the two columns before tokenizing each comment and training your Word2Vec model. 

In the cell below:

* Create a column called `'combined_text'` that consists of the data from the `'headline'` column plus a space character (`' '`) plus the data from the `'short_description'` column 
* Use the `'combined_text'` column's `.map()` method and pass in `word_tokenize`. Store the result returned in `data` 


```python
df['combined_text'] = None
data = None
```


```python
# __SOLUTION__ 
df['combined_text'] = df['headline'] + ' ' +  df['short_description']
data = df['combined_text'].map(word_tokenize)
```

Inspect the first 5 items in `data` to see how everything looks. 


```python
data[:5]
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
model = Word2Vec(data, size=100, window=5, min_count=1, workers=4)
```

Now, that you've instantiated Word2Vec model, train it on your text data. 

In the cell below:

* Call the `.train()` method on your model and pass in the following parameters:
    * The dataset we'll be training on, `data`
    * The `total_examples`  of sentences in the dataset, which you can find in `model.corpus_count` 
    * The number of `epochs` you want to train for, which we'll set to `10`


```python

```


```python
# __SOLUTION__ 
model.train(data, total_examples=model.corpus_count, epochs=10)
```




    (55564285, 67352790)



Great! You now have a fully trained model! The word vectors themselves are stored in the `Word2VecKeyedVectors` instance, which is stored in the `.wv` attribute. To simplify this, restore this object inside of the variable `wv` to save yourself some keystrokes down the line. 


```python
wv = None
```


```python
# __SOLUTION__ 
wv = model.wv
```

## Examining Your Word Vectors

Now that you have a trained Word2Vec model, go ahead and explore the relationships between some of the words in the corpus! 

One cool thing you can use Word2Vec for is to get the most similar words to a given word. You can do this by passing in the word to `wv.most_similar()`. 

In the cell below, try getting the most similar word to `'Texas'`.


```python

```


```python
# __SOLUTION__ 
wv.most_similar('Texas')
```




    [('Pennsylvania', 0.8078635931015015),
     ('Ohio', 0.7974496483802795),
     ('Louisiana', 0.7963728308677673),
     ('Arkansas', 0.7926678657531738),
     ('Oregon', 0.7917557954788208),
     ('Connecticut', 0.7912170886993408),
     ('Illinois', 0.79062819480896),
     ('Maryland', 0.7811000347137451),
     ('Massachusetts', 0.780368447303772),
     ('Georgia', 0.7769218683242798)]



Interesting! All of the most similar words are also states. 

You can also get the least similar vectors to a given word by passing in the word to the `.most_similar()` method's `negative` parameter. 

In the cell below, get the least similar words to `'Texas'`.


```python

```


```python
# __SOLUTION__ 
wv.most_similar(negative='Texas')
```




    [('End-of-School', 0.3979458212852478),
     ('Mother-in-law', 0.3947222828865051),
     ('Hunger-Free', 0.3708679676055908),
     ('Adulterers', 0.36885571479797363),
     ('Trashbag', 0.36644265055656433),
     ('once-reliable', 0.35787633061408997),
     ('teacher/student', 0.3503025770187378),
     ('Slammers', 0.34479519724845886),
     ('man-children', 0.3437780439853668),
     ('Veatch', 0.3415144681930542)]



This seems like random noise. It is a result of the way Word2Vec is computing the similarity between word vectors in the embedding space. Although the word vectors closest to a given word vector are almost certainly going to have similar meaning or connotation with your given word, the word vectors that the model considers 'least similar' are just the word vectors that are farthest away, or have the lowest cosine similarity. It's important to understand that while the closest vectors in the embedding space will almost certainly share some level of semantic meaning with a given word, there is no guarantee that this relationship will hold at large distances. 

You can also get the vector for a given word by passing in the word as if you were passing in a key to a dictionary. 

In the cell below, get the word vector for `'Texas'`.


```python

```


```python
# __SOLUTION__ 
wv['Texas']
```




    array([ 1.24636090e+00, -6.78244352e-01,  3.51518788e-03, -3.99456441e-01,
            4.69162464e-01, -3.62685633e+00,  2.67769009e-01,  3.79767013e-03,
           -1.01289642e+00, -7.34039128e-01,  5.45913994e-01,  5.65143108e-01,
           -1.50578868e+00,  6.64849877e-01, -2.79423738e+00, -1.56189334e+00,
            1.96586490e+00,  2.88979197e+00,  1.33330381e+00,  1.18798745e+00,
            7.27036059e-01,  1.01137400e+00,  8.20952579e-02,  1.19270813e+00,
            8.86384249e-01, -1.12821293e+00,  1.51647300e-01, -1.04708374e+00,
           -5.86788729e-02, -1.05820978e+00, -8.27524662e-01, -1.40232623e+00,
            6.86015546e-01, -4.46684033e-01,  1.24889505e+00, -1.32094190e-01,
           -5.03678262e-01, -9.22036409e-01,  1.10662019e+00, -5.96856654e-01,
            1.59698379e+00,  1.82010961e+00,  1.06715575e-01, -9.64384556e-01,
            6.02202535e-01,  5.30208468e-01,  3.87677163e-01, -9.98555362e-01,
            1.22688389e+00, -2.31141761e-01, -3.14787567e-01, -2.25427985e+00,
           -4.74881232e-01,  1.11975086e+00, -1.15890026e+00,  9.61482152e-03,
           -2.37981462e+00,  8.20012689e-01, -1.50626063e+00,  1.61282897e+00,
            1.21271707e-01, -6.15887642e-01,  1.58509552e-01, -3.96137571e+00,
            1.90922260e+00, -1.97405684e+00, -9.77061391e-01,  7.94807076e-01,
            8.33710492e-01, -2.87611216e-01,  5.39171733e-02,  1.66198456e+00,
            2.82799244e+00, -1.79123795e+00, -1.09240246e+00,  1.52354419e+00,
           -3.35005671e-01, -2.43378735e+00, -9.57983196e-01, -1.26319027e+00,
            1.52864003e+00, -8.02928209e-01, -3.59496951e-01, -1.32924289e-01,
            7.28161275e-01, -1.73035169e+00,  2.35984945e+00,  1.04702389e+00,
            4.83979806e-02,  1.48388743e+00,  5.28421640e-01,  2.44372666e-01,
            7.34018147e-01, -3.37905824e-01, -2.02289179e-01,  1.47428870e+00,
            2.15689039e+00, -1.01473296e+00, -3.01792049e+00,  6.76553249e-01],
          dtype=float32)



Now get all of the word vectors from the object at once. You can find these inside of `wv.vectors`. Try it out in the cell below.  


```python

```


```python
# __SOLUTION__ 
wv.vectors
```




    array([[ 2.4013439e-02,  3.9579192e-01, -1.5902083e+00, ...,
            -1.0319636e+00, -7.1263659e-01,  1.4866890e-01],
           [ 1.1389810e+00,  1.1225901e+00, -2.4732535e+00, ...,
             3.6903960e-01, -1.5361682e-01, -5.4053497e-01],
           [ 2.3889751e+00,  3.5420173e-01, -2.3829777e+00, ...,
            -2.5991669e-01, -4.5421821e-01,  1.8327081e+00],
           ...,
           [ 2.6314996e-02, -3.5944851e-03,  9.0951882e-02, ...,
             7.2956704e-02,  2.4217388e-02, -4.8497096e-02],
           [-9.7748280e-02,  8.0495318e-03,  5.6565467e-02, ...,
             2.7757246e-02,  2.9258214e-02, -4.0194180e-02],
           [ 2.3784293e-03,  2.1615300e-02,  6.6903256e-02, ...,
             1.0356646e-02,  6.6272311e-02, -4.3564435e-02]], dtype=float32)



As a final exercise, try to recreate the _'king' - 'man' + 'woman' = 'queen'_ example previously mentioned. You can do this by using the `.most_similar()` method and translating the word analogies into an addition/subtraction formulation (as shown above). Pass the original comparison, which you are calculating a difference between, to the negative parameter, and the analogous starter you want to apply the same transformation to, to the `positive` parameter.

Do this now in the cell below. 


```python

```


```python
# __SOLUTION__ 
wv.most_similar(positive=['king', 'woman'], negative=['man'])
```




    [('princess', 0.6285231113433838),
     ('brunette', 0.5955391526222229),
     ('reminiscent', 0.5760310888290405),
     ('queen', 0.5718123912811279),
     ('eccentricity', 0.5658714771270752),
     ('villain', 0.553656816482544),
     ('supermodel', 0.5518863201141357),
     ('title', 0.546796441078186),
     ('fan', 0.5466191172599792),
     ('diva', 0.5449496507644653)]



As you can see from the output above, your model isn't perfect, but 'Queen' and 'Princess' are still in the top 5. As you can see from the other word in top 5, 'reminiscent' -- your model is far from perfect. This is likely because you didn't have enough training data. That said, given the small amount of training data provided, the model still performs remarkably well! 

In the next lab, you'll reinvestigate transfer learning, loading in the weights from an open-sourced model that has already been trained for a very long time on a massive amount of data. Specifically, you'll work with the GloVe model from the Stanford NLP Group. There's not really any benefit from training the model ourselves, unless your text uses different, specialized vocabulary that isn't likely to be well represented inside an open-source model.

## Summary

In this lab, you learned how to train and use a Word2Vec model to create vectorized word embeddings!
