import os
import math
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

corpusroot = './US_Inaugural_Addresses'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
all_tokens = []

for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
            doc = file.read().lower() #converting into lowercase
            each_doc_tokens = tokenizer.tokenize(doc) #tokenising each doc 
            all_tokens.append(each_doc_tokens)

# stopword removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [[word for word in doc if word not in stop_words] for doc in all_tokens]

# stemming
stemmer = PorterStemmer()
stemmed_tokens = [[stemmer.stem(word) for word in doc] for doc in filtered_tokens]

df = defaultdict(int)  # token: df value
N = len(stemmed_tokens) # total number of docs in corpus

#get the df value for each (unique) token in each document and make the df dictionary
for doc in stemmed_tokens:
    unique_tokens = set(doc)  # unique tokens per document = non repeated 
    for token in unique_tokens:
        df[token] += 1  # increase that token's df value by 1 cause it appeared in the current doc 

# calculate the idf value for the given token
def getidf(token):
    token = stemmer.stem(token)  
    if token not in df:
        return -1
    return math.log10(N / df[token])  


