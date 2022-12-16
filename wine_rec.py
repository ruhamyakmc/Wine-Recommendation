import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel
import pickle
import regex
# read from folder dblib
df = pd.read_csv("dblib/winemag-data-130k-v2.csv").drop(["Unnamed: 0", "region_2"], axis=1)
# Load the data
def load_data(df):
    # predictor variables
    vars = [
        "country",
        "description",
        "designation",
        "province",
        "region_1",
        "title",
        "variety",
        "winery",
    ]
    predictors = df[vars].copy()
    predictors["name"] = predictors["designation"] + ", " + predictors["variety"]
    predictors.drop_duplicates(subset=["name"], keep="first", inplace=True)
    predictors.dropna(subset=["name"], inplace=True)
    predictors = predictors[predictors["country"] == "US"]
    predictors.reset_index(inplace=True)
    return predictors

predictors_final = load_data(df).copy()
# load the model
tfidf = TfidfVectorizer(
    stop_words="english",
    min_df=3,
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
)


def tfidf_matrix(predictors_final):
    tfidf_matrix = tfidf.fit_transform(predictors_final["description"])
    return tfidf_matrix


def sigmoid_kernel_s(tfidf_matrix):
    sigmoid_kernel_score = sigmoid_kernel(tfidf_matrix, tfidf_matrix)
    return sigmoid_kernel_score


def get_recommendations(name, sigmoid_kernel_score):
    # Get the original index corresponding
    index = pd.Series(
        predictors_final.index, index=predictors_final["name"]
    ).drop_duplicates()
    idx = index[name]
    sig_scores = list(enumerate(sigmoid_kernel_score[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:5]
    wine_indices = [i[0] for i in sig_scores]
    return predictors_final.iloc[wine_indices]


tfidf_matrix = tfidf_matrix(predictors_final)
sigmoid_kernel_score = sigmoid_kernel_s(tfidf_matrix)
#get_recommendations('Omira Hills, Merlot', sigmoid_kernel_score)

# Save the model
pickle.dump(sigmoid_kernel_score, open("sigmoid_kernel_score.pkl", "wb"))

if __name__ == "__main__":
    # load the model
    sigmoid_kernel_score = pickle.load(open("sigmoid_kernel_score.pkl", "rb"))
    # get the recommendations
    print(get_recommendations("Omira Hills, Merlot", sigmoid_kernel_score))



