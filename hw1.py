#!/usr/bin/env python
# coding: utf-8

# In[164]:


import os
import json
import boto3
import sklearn
import socket
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


print('user:', os.environ['JUPYTERHUB_SERVICE_PREFIX'])

def uiWebUrl(self):
    from urllib.parse import urlparse
    web_url = self._jsc.sc().uiWebUrl().get()
    port = urlparse(web_url).port
    return "{}proxy/{}/jobs/".format(os.environ['JUPYTERHUB_SERVICE_PREFIX'], port)

# small fix to enable UI views
SparkContext.uiWebUrl = property(uiWebUrl)

# spark configurtion in local regime 
conf = SparkConf().set('spark.master', 'local[*]').set('spark.driver.memory', '8g')

#some needed objects
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
spark


# # Part 1 Dataframe

# In[174]:


filepath = "file:///home/jovyan/shared/lectures_folder/84-0.txt"


# In[175]:


# read text as a dataframe
from pyspark.sql.functions import monotonically_increasing_id

dataframe= sc.textFile(f"{filepath}")     .map(lambda x: (x,))     .toDF()     .select(F.col("_1").alias("text"))     .withColumn("doc_id", monotonically_increasing_id())


# In[176]:


dataframe.show()


# In[177]:


dataframe.count()


# In[178]:


#function for filtering out non-letters, creating list of words
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
import string
import re

def process_string(data):
    non_letters_removed = re.sub(r'[^a-zA-Z]', ' ', data)
    words = non_letters_removed.lower().split(" ") 
    return list(filter(lambda x: len(x) > 0, words))
    
process_string_udf = udf(lambda z: process_string(z), ArrayType(StringType()))


# In[179]:


#filter out empty strings,apply function written above
by_words = dataframe    .select(process_string_udf(F.col("text")).alias("by_words"))    .where(F.size(F.col("by_words")) > 0)    .withColumn("doc_id", monotonically_increasing_id())


# In[180]:


by_words.show()


# In[181]:


#create a column with separate words in each row
by_words_count = by_words     .withColumn('word',(F.explode(F.col("by_words"))))


# In[182]:


by_words_count.show()


# In[183]:


from pyspark.sql.functions import count


# In[184]:


#calculate each term frequency in each document 
by_words_tf= by_words_count        .groupBy('doc_id', 'word').count()        .orderBy(F.col("doc_id"))        .withColumnRenamed("count", 'tf')


# In[185]:


by_words_tf.show()


# In[186]:


#calulate total number of docs
N_docs = by_words.count()
N_docs


# In[187]:


from pyspark.sql.functions import countDistinct


# In[188]:


#calculate document frequency
by_words_df = by_words_count       .groupBy("word")       .agg(countDistinct("doc_id").alias("df"))       .orderBy(F.col('df').desc())


# In[189]:


by_words_df.show()


# In[190]:


import numpy as np


# In[191]:


#function for calulating IDF
def calcIDF(df):
    IDF = np.log(N_docs/df)
    return IDF
calcIDF_udf = udf(lambda z: calcIDF(z).tolist())


# In[192]:


#apply function to the dataframe
by_words_idf = by_words_df       .withColumn('idf', calcIDF_udf(F.col("df")))


# In[193]:


by_words_idf.show()


# In[197]:


#calculate tf-idf for each word
by_words_tfidf = by_words_tf.select('doc_id', 'word', 'tf')       .join (by_words_idf, 'word')       .orderBy(F.col('doc_id').asc())       .withColumn("tf_idf", F.col("tf") * F.col("idf"))


# In[198]:


by_words_tfidf.show()


# In[199]:


#create tf-idf for each document(row), create final dataframe
by_words_final = by_words_tfidf       .groupBy("doc_id").sum("tf_idf")       .withColumnRenamed("sum(tf_idf)", 'tf_idf vector')


# In[200]:


by_words_final.show()


# In[201]:


#check that result is the dataframe
type(by_words_final)


# ## Part 2 RDD

# In[202]:


#read text as RDD
filepath = "file:///home/jovyan/shared/lectures_folder/84-0.txt"
from pyspark.sql.functions import monotonically_increasing_id

RDD = sc.textFile(f"{filepath}")

RDD.take(5)


# In[203]:


#lower letters, create a list of word in each row, delete empty rows
RDD_by_words = RDD       .map(lambda text: process_string(text))       .filter(lambda x: len(x)>0)


# In[204]:


RDD_by_words.take(10)


# In[205]:


#check if the number of rows equals to the result of part 1 calculation
if RDD_by_words.count() == N_docs:
    print('Numbers of rows are equal for both methods')
else:
    print('Numbers of rows differ by', RDD_by_words.count() - N_docs)


# In[206]:


#introduce indices for each document
RDD_by_words = RDD_by_words.zipWithIndex()


# In[207]:


RDD_by_words.take(5)


# In[208]:


#apply map function to list all instances of each word in the doc
RDD_tf = RDD_by_words.flatMap(lambda x: [((x[1],i),1) for i in x[0]])


# In[209]:


RDD_tf.take(10)


# In[210]:


#apply reduce function to count all instances of each word in each doc
RDD_tf=RDD_tf.reduceByKey(lambda x,y:x+y)


# In[211]:


RDD_tf.take(10)


# In[212]:


#recombine previous rdd to (word - (doc_id - tf))
RDD_2=RDD_tf.map(lambda x: (x[0][1],(x[0][0],x[1])))
RDD_2.take(5)


# In[213]:


#create an auxiliary rdd (word - 1), where each row is an occurence of a certain word in different documents
RDD_3=RDD_2.map(lambda x: (x[0],1))
RDD_3.take(5)


# In[214]:


#create rdd to calculate document frequency of each word
RDD_df=RDD_3       .reduceByKey(lambda x,y:x+y)


# In[215]:


RDD_df.takeOrdered(10, key = lambda x: -x[1])


# In[216]:


#calculate idf for each word
idf=RDD_df.map(lambda x: (x[0], np.log(N_docs/x[1])))
idf.take(10)


# In[217]:


idf.takeOrdered(10, key = lambda x: x[1])


# In[218]:


#auxiliary rdd, join 2 rdds, format of the final rdd (word - (doc_id - tf) - idf)
RDD_4 =RDD_2.join(idf)


# In[219]:


RDD_4.take(10)


# In[220]:


#auxiliary RDD to recombine rdd into format(doc_id - word - tf - idf - tf*idf)
RDD_5=RDD_4.map(lambda x: (x[1][0][0],(x[0],x[1][0][1],x[1][1],x[1][0][1]*x[1][1]))).sortByKey()
RDD_5.take(5)


# In[221]:


#auxiliary rdd, exclude all info apart from doc_id and tf-idf
RDD_6 = RDD_5.map(lambda x: (x[0], x[1][3]))
RDD_6.take(10)


# In[222]:


#final RDD with doc_id and tf-idf vector for each doc
RDD_final=RDD_6      .reduceByKey(lambda x,y:x+y)      .sortByKey()
RDD_final.take(10)


# In[223]:


#check that the type of output is RDD
type(RDD_final)

