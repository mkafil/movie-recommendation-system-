import os 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
# loading the data from the csv file to apandas dataframe        
os.chdir("D:\ds")
moviedata=pd.read_csv("movies.csv") 

# printing the first 5 rows of the dataframe
print(moviesdata.head())

# selecting the relevant features for recommendation
feat=['genres','keywords','tagline','cast','director']
# replacing the null valuess with null string
for feat in feat:
     moviedata[feat]=moviedata[feat].fillna('')

# combining all the 5 selected features
combdata=moviedata['genres']+''+moviedata['keywords']+''+moviedata['tagline']+''+moviedata['cast']+''+moviedata['director']

# converting the text data to feature vectors
vector=TfidfVectorizer()
featurevect=vector.fit_transform(combdata)

# getting the similarity scores using cosine similarity
similarity=cosine_similarity(featurevect)

# getting the movie name from the user
userinput='iron man'

# creating a list with all the movie names given in the dataset
title1=moviedata['title'].tolist()

# finding the close match for the movie name given by the user
closetitle=difflib.get_close_matches(userinput, title1)
closematch=closetitle[0]

# finding the index of the movie with title
indexva=moviedata[moviedata.title==closematch]['index'].values[0]

# getting a list of similar movies
similarityscore=list (enumerate(similarity[indexva]))

# sorting the movies based on their similarity score
sortedsimi=sorted(similarityscore, key= lambda x:x[1],reverse=True)
print('soretw')
print(sortedsimi)

# print the name of similar movies based on the index
print ('recommend')
i=1
for movie in sortedsimi:
    index=movie[0]
    titleofmovie=moviedata[moviedata.index==index]['title'].values[0]
    if(i<30):
      print (i,titleofmovie)
      i=i+1