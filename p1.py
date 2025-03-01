import os
import math
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# preprocessing definitions
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
corpusroot = './US_Inaugural_Addresses'

# global dictionaries and initialisation
df = defaultdict(int)
tf_idf_vectors = {}
N = 0  
posting_list = defaultdict(list)
global_filenames = []

# tokennise and preprocess given .txt file
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# calculate tf for a document 
def gettf(doc_tokens):
    tf = defaultdict(int)
    for word in doc_tokens:
        tf[word] += 1
    return tf

# calculate idf values for all the terms
def compute_idf():
    global N, df
    idf = {}
    for term, count in df.items():
        idf[term] = math.log10(N / count) if count > 0 else 0
    return idf

# calculates tf idf values for each document (and normalisation)
def compute_tfidf_vectors():
    global tf_idf_vectors, posting_list
    idf = compute_idf() # takes in idf values
    
    for doc_id, (filename, tokens) in enumerate(tf_idf_vectors.items()): # goes through all documents
        tf = gettf(tokens)
        tfidf_vector = {}
        norm = 0
        
        # computes tfidf values 
        for term, tf_value in tf.items():
            weight = (1 + math.log10(tf_value)) * idf.get(term, 0)
            tfidf_vector[term] = weight
            norm += weight ** 2
        
        #normalisation
        norm = math.sqrt(norm)
        for term in tfidf_vector:
            tfidf_vector[term] /= norm 
            posting_list[term].append((doc_id, tfidf_vector[term]))
        
        # sorts posting lists in descending order
        posting_list[term].sort(key=lambda x: x[1], reverse=True)
        tf_idf_vectors[filename] = tfidf_vector

# takes the input, performs stemming on it, then calls the func to calculate the idf values, 
# returns the idf value of a token
def getidf(token):
    token = stemmer.stem(token)
    return compute_idf().get(token, -1)

# takes the input, performs stemming on the token, then calls the func to calculate the tfidf values,
# returns the tfidf weight of the token in the given document
def getweight(filename, token):
    token = stemmer.stem(token)
    if filename not in tf_idf_vectors or token not in tf_idf_vectors[filename]:
        return 0
    return tf_idf_vectors[filename][token]

# given a query, returns the best matching doc (calculates cosine similarity)
def query(qstring):
    # preprocess the input query
    query_tokens = preprocess(qstring)

    # fetches individual tf and idf values
    query_tf = gettf(query_tokens)
    idf = compute_idf()
    
    query_vector = {}
    norm = 0

    # calculates tfdif values and normalisation
    for term, tf_value in query_tf.items():
        weight = (1 + math.log10(tf_value)) * idf.get(term, 0)
        query_vector[term] = weight
        norm += weight ** 2
    
    norm = math.sqrt(norm)
    for term in query_vector:
        query_vector[term] /= norm  
    
    # top 10 docs from the posting list and calcualte cos sim for each
    candidate_docs = defaultdict(float)
    for term in query_vector:
        for doc_id, weight in posting_list.get(term, [])[:10]:
            candidate_docs[doc_id] += query_vector[term] * weight
    
    if not candidate_docs:
        return ("fetch more", 0) # if not found, ask to fetch more than 10
    
    # highest cos sim = best matching doc
    best_doc_id = max(candidate_docs, key=candidate_docs.get)
    return (global_filenames[best_doc_id], candidate_docs[best_doc_id])

def main():
    """Main function to process corpus and handle queries."""
    global N, tf_idf_vectors, global_filenames
    
    # read and preprocess all .txt files
    for filename in os.listdir(corpusroot):
        if filename.endswith('.txt'):
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                content = file.read()
                tokens = preprocess(content)
                tf_idf_vectors[filename] = tokens
                for token in set(tokens):
                    df[token] += 1
                global_filenames.append(filename)
    
    N = len(tf_idf_vectors)
    compute_tfidf_vectors() # find tfidf weights
    
    # take user input search query and output the best matching document
    query_string = input("Enter the search query: ").strip()
    result = query(query_string)
    print(f"Best matching document in the corpus: {result[0]}, Score: {result[1]}")

    print("%.12f" % getidf('british'))
    print("%.12f" % getidf('union'))
    print("%.12f" % getidf('dollar'))
    print("%.12f" % getidf('constitution'))
    print("%.12f" % getidf('power'))
    print("--------------")
    print("%.12f" % getweight('19_lincoln_1861.txt','states'))
    print("%.12f" % getweight('07_madison_1813.txt','war'))
    print("%.12f" % getweight('05_jefferson_1805.txt','false'))
    print("%.12f" % getweight('22_grant_1873.txt','proposition'))
    print("%.12f" % getweight('16_taylor_1849.txt','duties'))
    print("--------------")
    print("(%s, %.12f)" % query("executive power"))
    print("(%s, %.12f)" % query("foreign government"))
    print("(%s, %.12f)" % query("public rights"))
    print("(%s, %.12f)" % query("people government"))
    print("(%s, %.12f)" % query("states laws"))
    
if __name__ == "__main__":
    main()
