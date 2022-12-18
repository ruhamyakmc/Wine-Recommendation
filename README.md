## Wine Recommendation System

This code implements a wine recommendation system that recommends wines based on their descriptions. The system uses a sigmoid kernel to calculate similarity scores between different wines, and recommends the most similar wines to a given wine.

The code will read in the winemag-data-130k-v2.csv file from the dblib folder, preprocess the data, and generate recommendations for the wine "Omira Hills, Merlot".

Saving and loading the model

The sigmoid kernel scores for the wines are saved to a pickle file called sigmoid_kernel_score.pkl. To load the model, you can use the following code:

`import pickle

sigmoid_kernel_score = pickle.load(open("sigmoid_kernel_score.pkl", "rb"))`
Customizing the recommendations

To get recommendations for a different wine, you can call the get_recommendations function with the name of the wine you want recommendations for and the sigmoid kernel scores:
'''
recommendations = get_recommendations("Name of Wine, Variety", sigmoid_kernel_score, predictors_final)
'''
The predictors_final dataframe contains information about the wines, including their names, descriptions, and other metadata. The get_recommendations function returns a dataframe with the recommended wines.

data source: https://www.kaggle.com/zynicide/wine-reviews

