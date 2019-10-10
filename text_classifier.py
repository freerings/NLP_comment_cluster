# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:12:45 2019

@author: Administrator
"""
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.externals import joblib

from nlp.text_clusters.tokenize_word import tokenize_and_stem, tokenize_only
from nlp.text_clusters.tokenize_word import train_tfidf


def get_origin_data():
    root_path = os.path.pardir(os.path.abspath(__file__))
    origin_file = os.path.join(os.path.join(root_path, "data_set/"), "comment_text.xls")
    origin_data = pd.read_excel(origin_file, sheet_name="document", encoding="utf-8")

    orig_sentence = origin_data["句子"]
    print(orig_sentence.head())
    return orig_sentence


def kmean_cluster(tfidf_matrix, num_cluster=5):
    km = KMeans(num_cluster)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    print(clusters)
    return km


def save_model(model, file_name="doc_cluster.pkl"):
    # use python bulit-in pickle package to save python object
    joblib.dump(model, file_name)
    
    # to reload the model and use it directly
    km = joblib.load(file_name)
    clusters = km.labels_.tolist()
    print(clusters)


if __name__ == "__main__":
    df = pd.DataFrame(get_origin_data())
    df["totalvab_stemmed"] = df["句子"].map(tokenize_and_stem)
    df["totalvab_tokenized"] = df["句子"].map(tokenize_only)
    
    print(df.head())
    
    tfidf_matrix = train_tfidf(get_origin_data().values)
    km = kmean_cluster(tfidf_matrix)
    save_model(km)
    df["cluster"] = km.labels_.tolist()
    df.index = [km.labels_.tolist()]
    df.to_excel("nlp_kmeans_out.xlsx", sheet_name="out", encoding="utf-8")
    
