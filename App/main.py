from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import pickle


app = FastAPI()


# =========================
# LOAD MODEL
# =========================

with open("Kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================
# LOAD SCALER
# =========================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================
# LOAD FEATURE NAMES
# =========================

with open("features_names.pkl", "rb") as file:
    feature_names = pickle.load(file)


# =========================
# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    return FileResponse("index.html")


# =========================
# PREDICTION
# =========================

@app.post("/predict")
def predict(

    latitude: float = Form(...),

    longitude: float = Form(...),

    confirmed: float = Form(...),

    deaths: float = Form(...),

    recovered: float = Form(...),

    active: float = Form(...),

    who_region: str = Form(...)

):

    # =========================
    # CREATE INPUT DATA
    # =========================

    input_data = {

        "Latitude": latitude,

        "Longitude": longitude,

        "Confirmed": confirmed,

        "Deaths": deaths,

        "Recovered": recovered,

        "Active": active

    }


    # =========================
    # WHO REGION ENCODING
    # =========================

    for feature in feature_names:

        if feature.startswith("WHO Region_"):

            region_name = feature.replace(
                "WHO Region_",
                ""
            )

            if region_name == who_region:

                input_data[feature] = 1

            else:

                input_data[feature] = 0


    # =========================
    # CREATE DATAFRAME
    # =========================

    input_df = pd.DataFrame([input_data])


    # =========================
    # SAME COLUMN ORDER
    # AS TRAINING DATA
    # =========================

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # =========================
    # SCALE INPUT
    # =========================

    input_scaled = scaler.transform(input_df)


    # =========================
    # PREDICT CLUSTER
    # =========================

    cluster = model.predict(input_scaled)[0]


    # =========================
    # SEND RESULT TO HTML
    # =========================

    return {
        "cluster": int(cluster)
    }