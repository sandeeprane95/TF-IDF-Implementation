## TF-IDF Implementation  

### Functionality of the code -  

1. The stopwords.txt file is being used for stopwords removal.  
2. Porter Stemmer from the NLTK library is used.  
3. parseStopwords function - Parses the stopwords.txt file and stores them in a list.  
4. tokenizer function - Converts each line read from a document into lower case and splits them on whitespaces.  
5. preprocessor function and documents reading loop - For each document, contents are read line by line and if the lines occur between <title></title> or <text></text> tags then they are tokenized. Then for each token, it is removed if it is a stopword, stemming is done and again checked to see if the stemmed token is a stopword. The punctuations and numbers are removed and if the token has <=2 characters, then it is removed. The docDict_list variable is a list of dictionaries, one for each document, that contains the tokens of the document and their frequencies.   
6. Indexing - The next part of the program creates an inverted index for the documents in the form of a list of dictionaries (tf_idf_matrix), one dictionary per document. Each dictionary contains the terms of the vocabulary as keys and their TF-IDF weights as values. TF-IDF matrix is a matrix of size m*n where m is the number of documents and n is the number of terms in the vocabulary. If a term doesn't exist in the document, then it TF-IDF value would be zero. In the program, if a term is not present in the document, then no key for that term exists in the TF-IDF dictionary of that document.  
7. Query parsing - The queries are parsed and preprocessed in a similar manner as the documents, by using the same tokenizer and preprocessor functions. Here we again calculate the tf_idf_matrix for the queries where tf is the term frequency of each term in a query whereas the idf is the same as the one produced in the indexing step.  
8. Cosine similarity function - Cosine similarity is used to find the similarity between each of the documents with each of the queries. The cosine similarity function takes a single number as a parameter and returns those many top ranked documents for each query.     
9. Avg. Precision and Avg. Recall computation - This part of the program uses the cosine similarity and relevance.txt document to find a precision and recall score for each query and then averages each of the score individually to produce the average precision score and average recall score for top 10, 50, 100 and 500 ranked documents for all the ten queries.         

### How to run the code -  

For Windows:  
1. Open Command prompt:   Start menu -> Run  and type 'cmd'. This will cause the Windows terminal to open.  
2. Type 'cd ' followed by project directory path to change current working directory, and hit Enter.   
3. Run the program using command 'python tf_idf.py'  
4. You will be prompted to input the paths to the dataset directory, stopwords file, queries file and relevance file.   
Alternatively, you can also use an IDE of your choice to execute the code.  

For Mac:  
1. Open Terminal by searching it using the search icon at top right or through 'Launchpad->Other->Terminal'   
2. Type 'cd ' followed by project directory path to change current working directory, and hit Enter.   
3. Run the program using command 'python3 tf_idf.py'  
4. You will be prompted to input the paths to the dataset directory, stopwords file, queries file and relevance file.  

## Output -   

Enter the path to the dataset directory/folder: ./cranfieldDocs/  

Enter the path to the stopwords file: ./stopwords.txt  

Enter the path to the queries file: ./queries.txt  

Enter the path to the relevance file: ./relevance.txt  

For top 10 documents in the ranking:  

Average Precision of all ten queries: 0.21000000000000002  
Average Recall of all ten queries: 0.19720760233918128  


For top 50 documents in the ranking:  

Average Precision of all ten queries: 0.1  
Average Recall of all ten queries: 0.42795321637426903  


For top 100 documents in the ranking:  

Average Precision of all ten queries: 0.06799999999999999  
Average Recall of all ten queries: 0.5384210526315789  


For top 500 documents in the ranking:  

Average Precision of all ten queries: 0.023600000000000003  
Average Recall of all ten queries: 0.9430555555555555  