from fastapi import FastAPI
import uvicorn
import wine_rec

app = FastAPI()
@app.get("/")
def hello():
    return "Welcome to my fast api"

# get the list of wine names from the predictors_final dataframe in wine_rec.py file and return a dropdown list of wine nam
@app.get("/wines")
def get_wines():
    return wine_rec.predictors_final["name"].values.tolist()[:20]

# get the list of wine names from the predictors_final dataframe in wine_rec.py file and return a dropdown list of wine name
@app.get("/recommend/{name}")
def recommend(name: str):
    return wine_rec.get_recommendations(name, wine_rec.sigmoid_kernel_score).to_dict("records")

if __name__ == "__main__":
    uvicorn.run(app, host="http://0.0.0.0", port=8080)

# Path: wine_rec.py

