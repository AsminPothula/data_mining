import os
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

# print total token count after tokenization
total_tokens = sum(len(doc) for doc in all_tokens)
print("Total tokens after tokenization:", total_tokens)

# stopword removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [[word for word in doc if word not in stop_words] for doc in all_tokens]

# print total token count after stopword removal
total_filtered_tokens = sum(len(doc) for doc in filtered_tokens)
print("Total tokens after stopword removal:", total_filtered_tokens)

# stemming
stemmer = PorterStemmer()
stemmed_tokens = [[stemmer.stem(word) for word in doc] for doc in filtered_tokens]

# print total token count after stemming
total_stemmed_tokens = sum(len(doc) for doc in stemmed_tokens)
print("Total tokens after stemming:", total_stemmed_tokens)
