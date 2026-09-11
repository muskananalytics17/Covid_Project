from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import pickle
import urllib.parse
import urllib.request
import json


app = FastAPI()


# =========================
# LOAD SAVED ML FILES
# =========================

with open("Kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("features_names.pkl", "rb") as file:
    feature_names = pickle.load(file)


# =========================
# CLUSTER INTERPRETATIONS
# Based on the cluster profiles
# from the COVID-19 training dataset.
# =========================

CLUSTER_ANALYSIS = {
    0: {
        "title": "High Case and Death Burden",
        "text": (
            "This input follows the pattern observed in Cluster 0. "
            "This cluster has a relatively high confirmed-case burden "
            "along with comparatively high deaths. It represents locations "
            "with a significant COVID-19 impact."
        )
    },
    1: {
        "title": "Extreme COVID-19 Burden",
        "text": (
            "This input follows the pattern observed in Cluster 1. "
            "This cluster has an extremely high number of confirmed, "
            "active and death cases compared with the other clusters. "
            "It represents the most severe case-burden pattern in the dataset."
        )
    },
    2: {
        "title": "Low Overall Case Burden",
        "text": (
            "This input follows the pattern observed in Cluster 2. "
            "This cluster has relatively low confirmed, death, recovered "
            "and active cases. It represents a comparatively low COVID-19 "
            "case-burden pattern."
        )
    },
    3: {
        "title": "Low Cases with Active Infections",
        "text": (
            "This input follows the pattern observed in Cluster 3. "
            "The overall confirmed and death counts are relatively low, "
            "but active cases form a noticeable part of the cluster profile. "
            "This indicates a lower overall burden with ongoing infections."
        )
    },
    4: {
        "title": "High Active Case Burden",
        "text": (
            "This input follows the pattern observed in Cluster 4. "
            "This cluster has relatively high confirmed cases and a "
            "substantial number of active cases. It represents a higher "
            "COVID-19 case-burden pattern with many infections still active."
        )
    },
    5: {
        "title": "High Cases with Strong Recovery Pattern",
        "text": (
            "This input follows the pattern observed in Cluster 5. "
            "The cluster has a high confirmed-case count and a relatively "
            "strong recovery pattern. Compared with some other high-case "
            "clusters, a larger share of cases is represented by recovered cases."
        )
    },
    6: {
        "title": "Very High Case and Active Burden",
        "text": (
            "This input follows the pattern observed in Cluster 6. "
            "The cluster has very high confirmed and active cases, together "
            "with substantial recovered and death counts. It represents a "
            "very high COVID-19 impact pattern."
        )
    }
}


# Average cluster values from the supplied COVID dataset
CLUSTER_PROFILE = {
    0: {"confirmed": 16548, "deaths": 1281, "recovered": 8192, "active": 7075},
    1: {"confirmed": 2066286, "deaths": 94990, "recovered": 766756, "active": 1204541},
    2: {"confirmed": 2551, "deaths": 90, "recovered": 1824, "active": 636},
    3: {"confirmed": 2415, "deaths": 49, "recovered": 1240, "active": 1126},
    4: {"confirmed": 15090, "deaths": 771, "recovered": 6807, "active": 7513},
    5: {"confirmed": 17912, "deaths": 465, "recovered": 11618, "active": 5829},
    6: {"confirmed": 24431, "deaths": 662, "recovered": 12853, "active": 10917}
}


# =========================
# LOCATION -> COORDINATES
# =========================

def get_coordinates(location: str):
    """
    Uses OpenStreetMap Nominatim to convert a location name
    into latitude and longitude.
    """
    query = urllib.parse.quote(location)

    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?q={query}&format=json&limit=1"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "COVID-19-Clustering-App"}
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results:
            return None, None

        return float(results[0]["lat"]), float(results[0]["lon"])

    except Exception:
        return None, None


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
    location: str = Form(...),
    confirmed: float = Form(...),
    deaths: float = Form(...),
    recovered: float = Form(...),
    active: float = Form(...)
):

    # -------------------------
    # GET LATITUDE & LONGITUDE
    # AUTOMATICALLY
    # -------------------------

    latitude, longitude = get_coordinates(location)

    if latitude is None or longitude is None:
        return {
            "success": False,
            "error": (
                "Location could not be found. "
                "Please enter a valid city, state or country."
            )
        }


    # -------------------------
    # CREATE INPUT DATA
    # -------------------------

    input_data = {
        "Lat": latitude,
        "Long": longitude,
        "Confirmed": confirmed,
        "Deaths": deaths,
        "Recovered": recovered,
        "Active": active
    }


    # -------------------------
    # CREATE DATAFRAME
    # -------------------------

    input_df = pd.DataFrame([input_data])


    # -------------------------
    # SAME COLUMN ORDER
    # AS TRAINING DATA
    # -------------------------

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # -------------------------
    # SCALE INPUT
    # -------------------------

    input_scaled = scaler.transform(input_df)


    # -------------------------
    # PREDICT CLUSTER
    # -------------------------

    cluster = int(model.predict(input_scaled)[0])


    # -------------------------
    # GET ANALYSIS
    # -------------------------

    analysis = CLUSTER_ANALYSIS[cluster]
    profile = CLUSTER_PROFILE[cluster]


    return {
        "success": True,
        "cluster": cluster,
        "title": analysis["title"],
        "analysis": analysis["text"],
        "profile": profile,
        "location": location,
        "latitude": latitude,
        "longitude": longitude
    }