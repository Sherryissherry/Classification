#!/usr/bin/env python
# coding: utf-8

# # Multi-Label Classification

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# # Dataset
# We will be using the CMU Movie Summary Corpus open dataset for this notebook. This dataset contains a list of movies and their genres. We can exploit movie summaries for predicting movie genres. 

# In[2]:


data_dir = '/dsa/data/DSA-8410/MovieSummaries/'


# In[3]:



meta = pd.read_csv(data_dir+"movie.metadata.tsv", sep = '\t', header = None)
meta.head()


# ## Set the proper column name for the dataframe.

# In[4]:


# rename columns
meta.columns = ["movie_id","freebase_movie_id","movie_name",
                "release_date","revenue","runtime", "languages","countries","genre"]
meta.head()


# # Load movie plots
# The movie plot is in a different file. We need to load the plot separately.

# In[5]:


plots = []

with open(data_dir + "plot_summaries.txt", 'r') as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            plots.append(row)
            
movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
  movie_id.append(i[0])
  plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})
movies.head()


# # Data Exploration and Pre-processing
# Now add the meta information to the movies dataframe. 

# In[6]:



# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movie plots
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

movies.head()


# In[7]:



movies['genre'][0]


# The tags are in json. We need to convert json to list

# In[8]:


import pandas as pd
import json

# Load the movies dataset
movies = pd.read_csv('/dsa/data/DSA-8410/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)

# Assign column names to the dataframe (if not already done)
movies.columns = ["movie_id", "freebase_movie_id", "movie_name", "release_date", "revenue", "runtime", "languages", "countries", "genre"]

# An empty list to store the genres
genres = []

# Extract genres
for i in movies['genre']:
    genres.append(list(json.loads(i).values()))

# Add the new genre list to the movies dataframe
movies['genre_new'] = genres


# In[9]:


movies.head()


# # T1. Drop movies which doesn't have any genre information
# 
# Dropping the movies which don't have any information about tags.

# In[10]:


# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
movies_new.shape, movies.shape


# # List all genres

# In[11]:


# get all genre tags in a list
all_genres = sum(genres,[])
len(set(all_genres))


# There are around 363 genres. This is too many. To reduce computing load, we will use top 50 gneres for prediciton. 
# 

# In[12]:


all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

g = all_genres_df.nlargest(columns="Count", n = 50) 

plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()


# In[13]:


selected_genre = list(g['Genre'])

# an empty list
tmp_genres = [] 

# extract genres
for i in movies['genre_new']: 
  tmp_genres.append(list(set(i).intersection(set(selected_genre)))) 

# add to 'movies' dataframe  
movies['chosen_genre'] = tmp_genres

movies.head()


# # T2. Drop rows that don't have any top-50 genres
# 
# We dropped the genres which are not in the top 50 list. So some movies now don't belong to any of these genres. We need to drop these movies. 

# In[14]:


# remove samples with 0 genre tags


# # T3. Clean the movie plot
# 
# This function drops the unnecessary characters from the movie plots. We will learn about regular expression later in this course. Use this function as a black box. 
# 

# In[15]:


# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


# Now, apply `clean_text` funciton in the dataframe to clean the plots.

# In[22]:


movies_new['clean_movie_name'] = movies_new['movie_name'].apply(lambda x: clean_text(x))


# # Check the clean plots now.

# All the top words are the stopwords, which won't help in predicting the movie tags. So we need to drop them. A python package named `nltk` has a stop words remover. We will use that to drop all the stopwords from the plots.

# In[23]:


movies_new.head()


# # Plot a frequency distribution of words in all the plots and identify the most frequent words. 

# In[29]:



if 'clean_movie_name' not in movies_new.columns:
    movies_new['clean_movie_name'] = movies_new['movie_name'].apply(lambda x: clean_text(x))


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # Select top n most frequent words
    d = words_df.nlargest(columns='count', n=terms)

    # Visualize words and frequencies
    plt.figure(figsize=(12,15))
    ax = sns.barplot(data=d, x='count', y='word')
    ax.set(ylabel='Word')
    plt.show()

# Print the 100 most frequent words
freq_words(movies_new['clean_movie_name'], 100)


# # Remove stop words
# 
# Most of the frequent words are stop words. We will download the list of stop words from `nltk` library and remove them from plots. 

# In[31]:


# download stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies_new['clean_movie_name'] = movies_new['clean_movie_name'].apply(lambda x: remove_stopwords(x))


# # Inspect the plots after removing the stopwords.

# In[32]:


movies_new.head()


# # Encoding target variables
# 
# We cannot use the text tags as targets directly in the model. We are required to convert the targets to multi-binary features. As we now have only 50 tags/genres, the number of target variables is 50. There is a 50-length output vector for each movie, where all the values will be zero except the corresponding movie tag position. 

# In[33]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['chosen_genre'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['chosen_genre'])


# In[34]:


y.shape


# # Convert text to feature vector
# 
# We can't train the model directly from the text. We need to convert it to a numeric vector feature. To convert the text to a feature vector, we will use sklearn's `TfidfVectorizer`. This method converts a text data to a numeric vector.  

# In[35]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=100)


# # T4. Create train (80%) and test (20%) split

# In[37]:


xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_movie_name'], y, 
                                             test_size=0.2, random_state=9)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# # T5. Multi-label Model Training
# 
# As we have multiple outputs (i.e., genres) for each movie, we will be using `MultiOutputClassifier` as it can learn multiple targets simultaneously. Internally it learns a model (aka base model) for every target. Let's use a decision tree classifier as a base model.

# In[40]:


from sklearn.model_selection import train_test_split

movies['clean_movie_name'] = movies['movie_name'].apply(clean_text)
print(movies.columns)
xtrain, xval, ytrain, yval = train_test_split(movies['clean_movie_name'], y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=100)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

clf = MultiOutputClassifier(DecisionTreeClassifier()).fit(xtrain_tfidf, ytrain)
y_pred = clf.predict(xval_tfidf)


# # T6. Measure Accuracy

# In[41]:


acc = np.sum(y_pred==yval) / (yval.shape[0]*yval.shape[1])
print(f"Acc: {acc:.2}")


# # T7. Qualitative evaluation: radnomly pick 10 plots, show their text, true genres, and predicted genres.

# In[44]:


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)



for i in range(10): 
    k = xval.sample(1).index[0] 
    print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", 
        infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")


# # Save your notebook, then `File > Close and Halt`
