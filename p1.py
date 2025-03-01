import os
import math
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Initialize tools
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
corpusroot = './US_Inaugural_Addresses'

# Global dictionaries
df = defaultdict(int)
tf_idf_vectors = {}
N = 0  # Total number of documents in corpus
posting_list = defaultdict(list)
global_filenames = []

def preprocess(text):
    """Tokenizes, removes stopwords, and stems words in text."""
    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

def compute_tf(doc_tokens):
    """Computes term frequency for a document."""
    tf = defaultdict(int)
    for word in doc_tokens:
        tf[word] += 1
    return tf

def compute_idf():
    """Computes inverse document frequency for all terms."""
    global N, df
    idf = {}
    for term, count in df.items():
        idf[term] = math.log10(N / count) if count > 0 else 0
    return idf

def compute_tfidf_vectors():
    """Computes and normalizes TF-IDF vectors for each document."""
    global tf_idf_vectors, posting_list
    idf = compute_idf()
    
    for doc_id, (filename, tokens) in enumerate(tf_idf_vectors.items()):
        tf = compute_tf(tokens)
        tfidf_vector = {}
        norm = 0
        
        for term, tf_value in tf.items():
            weight = (1 + math.log10(tf_value)) * idf.get(term, 0)
            tfidf_vector[term] = weight
            norm += weight ** 2
        
        norm = math.sqrt(norm)
        for term in tfidf_vector:
            tfidf_vector[term] /= norm  # Normalize
            posting_list[term].append((doc_id, tfidf_vector[term]))
        
        posting_list[term].sort(key=lambda x: x[1], reverse=True)
        tf_idf_vectors[filename] = tfidf_vector

def getidf(token):
    """Returns the IDF value of a token."""
    token = stemmer.stem(token)
    return compute_idf().get(token, -1)

def getweight(filename, token):
    """Returns the normalized TF-IDF weight of a token in a given document."""
    token = stemmer.stem(token)
    if filename not in tf_idf_vectors or token not in tf_idf_vectors[filename]:
        return 0
    return tf_idf_vectors[filename][token]

def query(qstring):
    """Processes a query and returns the best matching document based on cosine similarity."""
    query_tokens = preprocess(qstring)
    query_tf = compute_tf(query_tokens)
    idf = compute_idf()
    
    query_vector = {}
    norm = 0
    for term, tf_value in query_tf.items():
        weight = (1 + math.log10(tf_value)) * idf.get(term, 0)
        query_vector[term] = weight
        norm += weight ** 2
    
    norm = math.sqrt(norm)
    for term in query_vector:
        query_vector[term] /= norm  # Normalize query vector
    
    candidate_docs = defaultdict(float)
    for term in query_vector:
        for doc_id, weight in posting_list.get(term, [])[:10]:
            candidate_docs[doc_id] += query_vector[term] * weight
    
    if not candidate_docs:
        return ("fetch more", 0)
    
    best_doc_id = max(candidate_docs, key=candidate_docs.get)
    return (global_filenames[best_doc_id], candidate_docs[best_doc_id])

def main():
    """Main function to process corpus and handle queries."""
    global N, tf_idf_vectors, global_filenames
    
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
    compute_tfidf_vectors()
    
    query_string = input("Enter your search query: ").strip()
    result = query(query_string)
    print(f"Best matching document: {result[0]}, Score: {result[1]}")

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
