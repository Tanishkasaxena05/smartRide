
import joblib
import pandas as pd

MODEL_PATH = "models/demand_classifier.pkl"

_bundle = None


def load_bundle(path=MODEL_PATH):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def get_metadata():
    bundle = load_bundle()
    return {
        "model_name": bundle["model_name"],
        "features": bundle["features"],
        "classes": bundle["classes"],
        "slot_hours": bundle["slot_hours"],
        "slots_per_day": bundle["slots_per_day"],
        "n_zones": bundle["n_zones"],
        "low_threshold": bundle["low_threshold"],
        "high_threshold": bundle["high_threshold"],
        "zone_info": bundle["zone_info"],
        "zone_lookup": bundle["zone_lookup"],
        "split_date": bundle.get("split_date"),
        "comparison": bundle.get("comparison"),
    }


def zone_for_location(location):
    """Map a pickup location to its zone using the training-time assignment."""
    lookup = load_bundle()["zone_lookup"]
    match = lookup.loc[lookup["Pickup Location"] == location, "zone_id"]

    if match.empty:
        return None

    return int(match.iloc[0])


def predict_category(feature_dict, return_probabilities=False):
    """
    Parameters
    ----------
    feature_dict : dict keyed by feature name (see bundle["features"]).

    Returns
    -------
    str, or (str, dict) when return_probabilities is True.
    """
    bundle = load_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    features = bundle["features"]

    missing = [f for f in features if f not in feature_dict]
    if missing:
        raise KeyError(
            f"predict_category is missing required features: {missing}"
        )

    extra = [k for k in feature_dict if k not in features]
    if extra:
        raise KeyError(
            f"predict_category received features the model never saw: {extra}"
        )

    frame = pd.DataFrame([[feature_dict[f] for f in features]], columns=features)
    frame = frame.apply(pd.to_numeric, errors="coerce")

    encoded = model.predict(frame)[0]
    category = label_encoder.inverse_transform([encoded])[0]

    if not return_probabilities:
        return category

    probabilities = {}
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(frame)[0]
        probabilities = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(raw)
        }

    return category, probabilities
