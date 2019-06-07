
# coding: utf-8

# In[1]:


# NAME - Sandeep Rane 
#UIN - 677515266
# NETID - srane3


# In[2]:


import nltk, os, re, string, math
from nltk import PorterStemmer
#from nltk.corpus import stopwords
from pathlib import Path


# In[3]:


dirPath = 'cranfieldDocs'                                      # May be used if the user doesn't want to specify the files as separate input 
relevancePath = 'relevance.txt'
queriesPath = 'queries.txt'
stopwordPath = 'stopwords.txt'

dirPath = Path(input('Enter the path to the dataset directory/folder: '))
stopwordPath = Path(input('Enter the path to the stopwords file: '))
queriesPath = Path(input('Enter the path to the queries file: '))
relevancePath = Path(input('Enter the path to the relevance file: '))


# In[4]:


stopword_list = []
docDict_list = []
doc_count = 0
vocabulary = {}
vocabSize = -1
docFreq = {}
total_term_in_doc = []
term_max_tf = []


# In[5]:


def parseStopwords():
    file_object = open(stopwordPath, 'r')
    for aStopword in file_object:
        aStopword = aStopword.lower()
        aStopword = re.split("\n",aStopword)
        stopword_list.append(aStopword[0])
    file_object.close()


# In[6]:


parseStopwords()
#stopword_list = set(stopwords.words('english'))


# In[7]:


def tokenizer(file_content):
    file_content = file_content.lower()
    generatedTokens = file_content.split()
    return generatedTokens


# In[8]:


stemmer = PorterStemmer()


# In[9]:


def preprocessor(aToken):
    if aToken not in stopword_list:
        exclude = set(string.punctuation)
        new_s = ''.join(ch for ch in aToken if ch not in exclude)
        aToken = ''.join([i for i in new_s if not i.isdigit()])
        stemWord = stemmer.stem(aToken)
        if stemWord not in stopword_list:
            return stemWord
        return 'x'                        #Return a string which would be eliminated 
    return 'x'                            #Return a string which would be eliminated


# In[10]:


for file_x in os.listdir(dirPath):
    file_object = dirPath / file_x
    file_content = file_object.read_text() 
    maxTermFreq = 1
    termCount = 0
    docDict = {}
    tokens = tokenizer(file_content)
    reqd_token = False
    for aToken in tokens:
        if aToken in ['<title>','<text>']:
            reqd_token = True
            continue
        if aToken in ['</title>','</text>']:
            reqd_token = False
            continue
        if reqd_token is True:
            newToken = preprocessor(aToken)
            if not (len(newToken)<3):
                termCount += 1
                if newToken not in vocabulary:
                    vocabSize += 1
                    vocabulary[newToken] = vocabSize
                    docFreq[newToken] = 0
                if newToken not in docDict:
                    docDict[newToken] = 1
                    docFreq[newToken] += 1
                else:
                    docDict[newToken] = docDict[newToken] + 1
                    if (docDict[newToken]>maxTermFreq):
                        maxTermFreq = docDict[newToken]
    term_max_tf.append(maxTermFreq)
    docDict_list.append(docDict)
    total_term_in_doc.append(termCount)
    doc_count += 1


# In[11]:


tf_idf_matrix = []
term_idf = {}
doc_vec_len = []
for j in range(doc_count):
    new_row = {}
    total = 0
    for k,v in docDict_list[j].items():
        tf_val = float(v)/term_max_tf[j]
        idf_val = math.log((float(doc_count)/docFreq[k]),2)
        term_idf[k] = idf_val
        tf_idf_val = tf_val*idf_val
        total += (tf_idf_val**2)
        new_row[k] = tf_idf_val
    tf_idf_matrix.append(new_row)
    doc_vec_len.append(total**0.5)


# In[12]:


query_object = open(queriesPath, 'r')
query_list = []
query_max_tf = []
for query in query_object:
    tokens = tokenizer(query)
    query_vocab = {}
    maxTermFreq = 1
    for aToken in tokens:
        newToken = preprocessor(aToken)
        if not (len(newToken)<3):
            if newToken not in query_vocab:
                query_vocab[newToken] = 1
            else:
                query_vocab[newToken] += 1
                if query_vocab[newToken]>maxTermFreq:
                    maxTermFreq = query_vocab[newToken]
    query_list.append(query_vocab)
    query_max_tf.append(maxTermFreq)
query_object.close()


# In[13]:


query_tfidf_matrix = []
i = 0
query_vec_len = []
for query_dict in query_list:
    new_dict = {}
    total = 0
    for k in query_dict.keys():
        q_tf = query_dict[k]/query_max_tf[i]
        if k in term_idf:
            idf = term_idf[k]
        else:
            idf = 0
        q_tfidf = q_tf*idf
        total += (q_tfidf**2)
        new_dict[k] = q_tfidf
    query_tfidf_matrix.append(new_dict)
    query_vec_len.append(total**0.5)


# In[14]:


cos_result = []
def cosine_sim(top_x):
    for q_no in range(len(query_vec_len)):
        query_result = []
        for doc_no in range(doc_count):
            total = 0
            for k,v in query_tfidf_matrix[q_no].items():
                if k in tf_idf_matrix[doc_no]:
                    total += (tf_idf_matrix[doc_no][k]*v)
            total /= (query_vec_len[q_no]*doc_vec_len[doc_no])
            if(total>0):
                query_result.append((total,doc_no+1,q_no+1))
        query_result.sort(reverse=True)
        cos_result.append(query_result[0:top_x])


# In[15]:


cosine_sim(500)


# In[16]:


topN = [10, 50, 100, 500]
relevance_object = open(relevancePath, 'r')
relevance_list = []
for i in range(len(query_vec_len)):
    relevance_list.append([])
for rel_data in relevance_object:
    gs_data = rel_data.split()
    relevance_list[int(gs_data[0])-1].append(int(gs_data[1]))
relevance_object.close()


# In[17]:


for n in topN:
    print("\nFor top "+ str(n) +" documents in the ranking:\n")
    p_total = 0
    r_total = 0
    for q_no in range(len(query_vec_len)):
        rp_num = 0        
        for retrievedDoc in cos_result[q_no][:n]:
            if retrievedDoc[1] in relevance_list[q_no]:
                rp_num += 1
        recall_n = float(rp_num)/len(relevance_list[q_no])
        precision_n = float(rp_num)/(n)
        p_total += precision_n
        r_total += recall_n
    avg_p = float(p_total)/len(query_vec_len)
    avg_r = float(r_total)/len(query_vec_len)
    print("Average Precision of all ten queries: " + str(avg_p))
    print("Average Recall of all ten queries: " + str(avg_r) + "\n")

