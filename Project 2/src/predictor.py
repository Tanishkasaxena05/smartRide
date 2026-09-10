import joblib
import pandas as pd


model = joblib.load(
    "models/demand_classifier.pkl"
)


weekday_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


def predict_category(
    hour,
    weekday,
    previous_demand,
    lat,
    lon
):

    df = pd.DataFrame(
        [[
            hour,
            weekday_map[weekday],
            previous_demand,
            lat,
            lon
        ]],
        columns=[
            "hour",
            "weekday_encoded",
            "previous_demand",
            "lat",
            "lon"
        ]
    )

    return model.predict(df)[0]