import pandas as pd
import numpy as np
from math import log
import spacy
import os
from numpy.linalg import svd
pd.options.mode.chained_assignment = None

MODEL_NAME="en_core_web_md"

FILE_PATH=input("Path to .txt file : ")
SUMMARY_SIZE=input("Summary size (percentage): ")

def computeSVD(x):
    """
    Computes the SVD of the given matrix
    AV=US
    
    :param x: numpy array
    :return: U,S,V.T
    """
    u,s,v=svd(x)
    e=list(s)
    s=np.zeros(x.shape)
    for i in range(len(e)):
        s[i][i]=e[i]
    return u,s,v

def summarizer(filePath,summarySize):
    """
    Creates the summary of the given text

    :param filePath: path to .txt file
    :param summarySize: desired summary size (percent of original text)
                        default size = 40

    :return: creates a .txt file
    """
    global MODEL_NAME
    
    if summarySize=='':
        summarySize=0.4
    else:
        summarySize=int(summarySize)/100
    
    # load the text file
    with open(filePath,'r',encoding='utf-8') as f:
        text=f.read().split('\n')
        text=' '.join(text).strip()

    # load the spacy model
    model=spacy.load(MODEL_NAME)

    # create tokens 
    tokens=model(text)
    tokens=[t for t in tokens if (not t.is_stop) and (not t.is_punct)]
    tokens=[str(t.lemma_).lower() for t in tokens if not t.is_space]
    tokens=list(set(tokens))

    sentences=[]
    doc=[]
    sent=''

    for w in text.split():

        sent=sent+" "+w
        if (w[-1]=='.') and (not w[-2].isupper()):
            doc.append(sent.strip())
            sent_tokens=model(sent)
            sent_tokens=[str(t.lemma_).lower() for t in sent_tokens if not t.is_punct]
            sent=' '.join(sent_tokens).strip()
            sentences.append(sent[:])
            sent='' 

    # create dataframe to create TF-IDF input matrix
    df=pd.DataFrame(columns=['sent'+str(s) for s in range(len(sentences))],index=tokens)
    df.fillna(0,inplace=True)

    # TF-IDF

    # compute sentence frequency and inverse sentence frequency (sf and isf) for each sentence
    sf=0
    tfidf=[]
    for word in df.index:
        for i,sent in enumerate(sentences):
            if word in sent.split():
                sf+=1

        isf=log(len(sentences)/sf + 1)

        # compute word frequency for each word in each sentence
        # and get the tfidf score
        for i,sent in enumerate(sentences):
            if word in sent.split():
                wf=sent.split().count(word)/len(sent.split())
                df['sent'+str(i)].loc[word]=wf*isf 
                tfidf.append(wf*isf)

    # create input matrix and get the svd
    A=df.values
    u,s,v=computeSVD(A)

    # get average score and replace the scores with 0 
    # if score is less than equal to average
    for i in range(len(v.T)):
        row=v.T[i]
        mean=sum(row)/len(row)
        for j,val in enumerate(row):
            if val <= mean: 
                row[j]=0

    # get sentences for summary, with the top score sum
    sent_scores={}
    for i in range(len(v)):
        sent_scores[sum(v[i])]=i
    num=int(np.round(len(doc)*summarySize))      
    top=sorted(list(sent_scores))[-num:]

    # print the summary
    summary=""
    for score in top:
        i=sent_scores[score]
        summary=summary+" "+doc[i]
    summary=summary.strip()

    filename=os.path.basename(filePath).split('.')[0]
    output_file_name=filename+'_summarized_'+str(int(summarySize*100))+".txt"
    
    with open(output_file_name,"w") as f:
        f.write(summary)
        
    print("\nSummary created !!")

summarizer(FILE_PATH,SUMMARY_SIZE)
