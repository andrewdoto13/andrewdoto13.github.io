---
title: "Random Walks on a Markov Chain constructed from Text Messages"
classes: wide
tags: [python, markov chain, artificial intelligence]
desc: "Using text messages as a corpus to construct Markov Chains to auto-generate texts"
---

In the Winter of 2021, I took a course called "Elements of Artificial Intelligence" as part of my data science master's program. One of my favorite modules from the course was on Hidden Markov models, and Markov Chains were a part of that module. I think I really enjoyed its simplicity and applicability; they are easy to understand and I thought that it might be fun to play with Markov Chains for a small side project. I was particularly inspired by one of the slides Prof. Crandall presented, where he detailed this project where some researchers had constructed a Markov Chain from posts on an early-internet version of a dating website. They were able to do "random walks" along this Markov Chain to produce automatically-generated posts. The results sort-of made sense but not quite, so the sentences in these generated posts were just hilarious.

Taking that idea, I thought that I could apply this concept of having some corpus of text and constructing a Markov Chain from it. We had an assignment during the class that had us develop an implementation of a Hidden Markov Model, so it wouldn't be too difficult to tweak slightly for a small side project. I came up with the idea to use text messages from my family and friends as the corpus of text to construct the Markov Chain's transition and initial state probabilities.

Before I delve into the implementation, I'll give a quick high-level overview of the project. Essentially, what I'm doing is taking one person's phone number that I've texted with and execute a query from the Messages application database on my Macbook to pull their texts. The SQL is very simple; all you really have to know is how to join two tables properly, which I easily found by exploring the database using SQLite. Then once I have the texts, I import them into dictionaries and pandas dataframes to construct initial state and transition probabilities. Having these two sets of probability distributions (which represent what you need for a Markov Chain) then allows me to write some simple functions to do a random walk based on these probabilities to then construct "text messages."

**A few quick notes**

* There are a few ways where I'd like to eventually optimize the code, namely, using Numpy-based operations to construct the probabilities
* There's definitely some clean-up to the text messages that I could do. For example, when somebody does a "Tapback," (it's when somebody "reacts" to a text inside the Messages app), it's recorded in the database like "'Laughed at \[text].'"
* I will eventually go back and "clean up" the code; this is just meant to be more hacky and fun. My family and I have been getting some good laughs out of it, so from that standpoint, it's been mission-accomplished.

With that said, I'd like to walk through the implementation!

To start, I'll import a few libraries, and then use sqlite to connect to the Messages database on my MacBook.


```python
import sqlite3
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
conn = sqlite3.connect('chat.db')
phone_number = 'phone_number'
```

This next cell here contains the query I used to pull the texts; I've removed the phone number, but for this example I'm using texts from my sister-in-law Kathleen as the corpus (I don't think she will mind!). Then I used pandas to execute the query and load the data into a DataFrame.


```python
query = '''
select
    *
from
    handle handle
    inner join
    message message
        on
        handle.rowid = message.handle_id
where
    handle.id = ''' + "'" + phone_number + "'"

texts = pd.read_sql(query, conn)
```

Now that the texts are loaded into a DataFrame, I want to filter them so that I'm only using texts that are from Kathleen to me. I could have added this into the WHERE clause in the SQL as well.


```python
texts = texts.loc[
    texts.is_from_me == 0
]
```

Then what I chose to do is to basically take the texts and split them so that each word is one element in a list. The lengthy second line of code is basically ensuring that each element is truly a list. I believe that I ran into a few issues where some texts weren't converting properly into lists, so this was my way of filtering them out. One final thing that I added that is important to the Markov Chain is that I added an END_TEXT element to the end of each list for each text. This was my way of building in something so that the Markov Chain would "find" the end of a "text" during a random walk.


```python
texts = texts.text.str.strip()

texts = texts.str.split().loc[
    texts.str.split().apply(lambda x: type(x)).astype(str) == "<class 'list'>"
    ].apply(lambda x: x + ['END_TEXT'])
```


```python
len(texts)
```




    816



Now that I have the texts formatted the way I need, I can now go ahead and start building the probability distributions. This next section of code is where I built the initial state distribution. What you'll see is that I've taken all the first words in a text, did a frequency count, and then normalized them by the total. In index 4 and 5 of this DataFrame you'll see "Loved" and "Laughed," these are the pesky Tapbacks that I mentioned earlier. Having this distribution is key because I can use this to start the texts off in a random walk.


```python
texts = texts.apply(lambda x: np.array(x))
init = pd.DataFrame(texts.apply(lambda x: x[0]).value_counts() / texts.apply(lambda x: x[0]).value_counts().sum()).reset_index()
init.columns = ['word', 'prob']
init.sort_values(by = 'prob', ascending = False).head(10)
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
      <th>word</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>￼</td>
      <td>0.101716</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I</td>
      <td>0.099265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I’m</td>
      <td>0.025735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>It’s</td>
      <td>0.024510</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Loved</td>
      <td>0.023284</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Laughed</td>
      <td>0.020833</td>
    </tr>
    <tr>
      <th>6</th>
      <td>He</td>
      <td>0.019608</td>
    </tr>
    <tr>
      <th>7</th>
      <td>That’s</td>
      <td>0.018382</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Yeah</td>
      <td>0.017157</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What</td>
      <td>0.013480</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Checking that the probabilities add correctly: ", init.prob.sum())
```

    Checking that the probabilities add correctly:  0.9999999999999999


These next two code blocks are where most of the work was. This first block here is where I've looped through all the texts and built the transition probabilities, which take the form of a nested dictionary. All that's happening here is that I'm looking at each word and the word after each word in all of the texts, counting them up, and placing them into their dictionaries. I then normalize them to show probabilities instead of the frequency counts. The dictionary I printed shows the probabilities of the transitions from the word "I"; so you can see that another dictionary is nested inside, with each word after "I" as the keys and the values as the transition probability to that word. For the sake of avoiding printing the entire dictionary, I've placed it into a DataFrame and showed only the first few rows.


```python
tr = {}
word_counts = {}

for sentence in texts:
    for placement, word in enumerate(sentence):
        if word not in tr.keys():
            tr[word] = {}
            word_counts[word] = 0
        if word == "END_TEXT":
            break
        word_after = sentence[placement+1]
        if word_after not in tr[word].keys():
            tr[word][word_after] = 1
        else:
            tr[word][word_after] += 1
        word_counts[word] += 1

for word in tr.keys():
    for word_after in tr[word].keys():
        if word_counts[word] > 0:
            tr[word][word_after] = tr[word][word_after] / word_counts[word]
```


```python
pd.DataFrame({'word':tr['I'].keys(), 'prob':tr['I'].values()}).head()
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
      <th>word</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>know</td>
      <td>0.037915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>won't</td>
      <td>0.004739</td>
    </tr>
    <tr>
      <th>2</th>
      <td>think</td>
      <td>0.052133</td>
    </tr>
    <tr>
      <th>3</th>
      <td>could</td>
      <td>0.033175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>love</td>
      <td>0.047393</td>
    </tr>
  </tbody>
</table>
</div>



This next block has the function definitions for doing the random walk along the Markov Chain. The key to these functions is that I'm placing the transition probabilities into a pandas DataFrame and using a provided method to do a random sample for words weighted by the probabilities. Now that I have all the elements I need, I can simply call the functions to auto-generate texts that sound like Kathleen! Below I've printed 10 texts as examples. As you can see, the texts *kinda* make sense, but because of the naive nature of Markov Chains, they aren't truly coherent. But that is the fun part after all!


```python
def gen_first_word():
    return init.sample(1, weights = init['prob'], replace = True)['word'].iat[0]

def gen_next_word(word):
    return pd.DataFrame({'word': tr[word].keys(),
                         'prob':tr[word].values()}).sample(1, weights = init['prob'], replace = True)['word'].iat[0]

def gen_sentence():
    first_word = gen_first_word()
    sentence = first_word + " "
    word = first_word
    while word != 'END_TEXT':
        word = gen_next_word(word)
        if word != 'END_TEXT':
            sentence = sentence + word + ' '
    return sentence
```


```python
for i in range(10):
    print(gen_sentence(), '\n')
```

    Cool thanks  

    It was such a shot with rice noodles  

    You’re already married to their sale for ps4?  

    It was quite a video of stairs or the professionals if I’m thinking that Andy’s Ugly Christmas day right?  

    Looks good!! I was delicious  

    Ok  

    Did you just talk to Corvallis!  

    Very busy but still use the best part!!  

    What kind of something similar to everyone prove they don’t sleep lol 😂  

    Hmmm I know what a Slytherin lol 😆  



And that's it! Thank you for taking the time to read through this side-project of mine. I'd love to hear your thoughts and feedback, so never hesitate to reach out. Take care!
