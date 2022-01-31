# Exctractive text summarization
Exctractive text summarization is an unsupervised approach that uses LSA (Latent Semmantic Analysis) with SVD (Single Value Decomposition) to determine the importance score of each document (or sentence) with respect to the context. Based on the scores, it picks the top most important sentences for the summary.
Refer to https://www.researchgate.net/publication/220195824 for detail information

# LSA - Latent Semantic Analysis 
It's a technique to analyze relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms.
LSA is an unsupervised means of learning

# SVD - Single Value Decomposition
It is a dimension reduction algorithm. It breaks a `m x n` matrix into 3 special matrix that are easy to analyze
![image](https://user-images.githubusercontent.com/77575222/151765221-a0fa255e-f6d5-4795-86e4-59c76b9a6bde.png)

# Prerequisite
Install the dependencies
`pip install -r requirements.txt`

Download the spacy model
`python -m spacy download en_core_web_md`

# Usage
1. Navigate to the repository through command prompt
2. Type `python summarizer.py`
3. Enter the file path to the .txt file that contains the text
4. Provide the size of summary (percentage value). Press enter to use default value (default value = 40)
5. Summary text file is generated as `<name of the text file>_summarised_<summary size>.txt`

