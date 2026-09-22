import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =============================================================
# 1. LOAD DATA
# =============================================================

input_path = "data/ncr_ride_bookings.csv"

# If the file isn't found in data/, try the same directory
# as this Python file.
if not os.path.exists(input_path):
    input_path = "ncr_ride_bookings.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(
        "Could not find 'ncr_ride_bookings.csv'.\n\n"
        "Make sure your file is located at:\n"
        "data/ncr_ride_bookings.csv\n\n"
        "or in the same directory as this Python file."
    )


# Load CSV
df = pd.read_csv(input_path)

print("=" * 70)
print("NCR RIDE BOOKING - HIERARCHICAL CLUSTERING")
print("=" * 70)

print(f"\nDataset loaded successfully.")
print(f"Input file : {input_path}")
print(f"Rows       : {df.shape[0]:,}")
print(f"Columns    : {df.shape[1]}")


# =============================================================
# 2. CLEAN COLUMN NAMES

# Remove accidental leading/trailing spaces from column names.
df.columns = df.columns.str.strip()

print("\nAvailable columns:")
for column in df.columns:
    print(f"  {column}")


# =============================================================
# 3. SELECT FEATURES FOR CLUSTERING


# These names exactly match the columns in your CSV.
candidate_features = [
    "Booking Value",
    "Ride Distance",
    "Driver Ratings",
    "Customer Rating",
    "Avg VTAT",
    "Avg CTAT",
]


# Only use columns that actually exist.
features = [
    column
    for column in candidate_features
    if column in df.columns
]

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print("\nRequested clustering features:")
for feature in candidate_features:
    print(f"  {feature}")

print("\nFeatures found in dataset:")
for feature in features:
    print(f"  ✓ {feature}")


# =============================================================
# 4. CHECK THAT FEATURES EXIST
# =============================================================

if len(features) == 0:
    print("\nERROR: No clustering features were found.")

    print("\nYour CSV contains:")
    print(df.columns.tolist())

    raise ValueError(
        "\nNo valid clustering features were found. "
        "Check the CSV column names."
    )


# =============================================================
# 5. CONVERT FEATURES TO NUMERIC
# =============================================================

print("\nConverting clustering features to numeric values...")

for feature in features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# =============================================================
# 6. REMOVE COMPLETELY EMPTY FEATURES
# =============================================================

valid_features = []

for feature in features:

    valid_count = df[feature].notna().sum()

    print(
        f"{feature}: "
        f"{valid_count:,} valid values"
    )

    if valid_count > 0:
        valid_features.append(feature)
    else:
        print(
            f"WARNING: {feature} contains no usable numeric data."
        )


features = valid_features

print("\nFinal features used for clustering:")
for feature in features:
    print(f"  ✓ {feature}")


if len(features) == 0:
    raise ValueError(
        "None of the selected features contain usable numeric data."
    )


# =============================================================
# 7. CREATE FEATURE MATRIX
# =============================================================

X = df[features].copy()

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

print(
    f"\nFeature matrix shape: {X.shape}"
)

print("\nMissing values before imputation:")

for feature in features:

    missing = X[feature].isna().sum()

    print(
        f"  {feature}: {missing:,}"
    )


# =============================================================
# 8. IMPUTE MISSING VALUES
# =============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)

print(
    f"Imputed matrix shape: {X_imputed.shape}"
)


# =============================================================
# 9. STANDARDIZE FEATURES
# =============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

print(
    f"Scaled matrix shape: {X_scaled.shape}"
)


# =============================================================
# 10. CLUSTERING SETTINGS
# =============================================================

N_CLUSTERS = 4

# HCA can become extremely memory intensive on large datasets.
MAX_DIRECT_ROWS = 15000

# Sample size for hybrid HCA.
MAX_SAMPLE_SIZE = 5000


if len(df) < N_CLUSTERS:

    raise ValueError(
        f"The dataset contains only {len(df)} rows, "
        f"but {N_CLUSTERS} clusters were requested."
    )


# =============================================================
# 11. CREATE OUTPUT DIRECTORY
# =============================================================

output_directory = "data/cleaned"

os.makedirs(
    output_directory,
    exist_ok=True
)


# =============================================================
# 12. DIRECT HIERARCHICAL CLUSTERING
# =============================================================

if len(df) <= MAX_DIRECT_ROWS:

    print("\n" + "=" * 70)
    print("DIRECT HIERARCHICAL CLUSTERING")
    print("=" * 70)

    print(
        f"\nDataset contains {len(df):,} rows."
    )

    print(
        f"Direct HCA limit is {MAX_DIRECT_ROWS:,} rows."
    )

    print(
        "\nRunning hierarchical clustering directly..."
    )


    # ---------------------------------------------------------
    # Create dendrogram
    # ---------------------------------------------------------

    print("\nCreating dendrogram...")

    linked = linkage(
        X_scaled,
        method="ward",
        metric="euclidean"
    )

    plt.figure(
        figsize=(12, 6)
    )

    dendrogram(
        linked,
        truncate_mode="lastp",
        p=30,
        show_contracted=True
    )

    plt.title(
        "Hierarchical Clustering Dendrogram"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Distance"
    )

    dendrogram_path = os.path.join(
        output_directory,
        "dendrogram.png"
    )

    plt.savefig(
        dendrogram_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Dendrogram saved to:\n"
        f"{dendrogram_path}"
    )


    # ---------------------------------------------------------
    # Run Agglomerative Clustering
    # ---------------------------------------------------------

    print("\nRunning Agglomerative Clustering...")

    hca = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        linkage="ward"
    )

    cluster_labels = hca.fit_predict(
        X_scaled
    )

    df["cluster"] = cluster_labels

    print(
        "Direct HCA completed successfully."
    )


# =============================================================
# 13. HYBRID HCA FOR LARGE DATASETS
# =============================================================

else:

    print("\n" + "=" * 70)
    print("HYBRID HIERARCHICAL CLUSTERING")
    print("=" * 70)

    print(
        f"\nDataset contains {len(df):,} rows."
    )

    print(
        f"Direct HCA limit is {MAX_DIRECT_ROWS:,} rows."
    )

    print(
        "\nDataset is too large for direct HCA."
    )

    print(
        "Using sample-based HCA instead."
    )


    # ---------------------------------------------------------
    # Determine sample size
    # ---------------------------------------------------------

    sample_size = min(
        MAX_SAMPLE_SIZE,
        len(df)
    )

    print(
        f"\nSample size: {sample_size:,}"
    )


    # ---------------------------------------------------------
    # Reproducible random sample
    # ---------------------------------------------------------

    np.random.seed(42)

    sample_indices = np.random.choice(
        len(df),
        size=sample_size,
        replace=False
    )

    X_sample_scaled = X_scaled[
        sample_indices
    ]


    # =========================================================
    # 13A. SAMPLE DENDROGRAM
    # =========================================================

    print("\nCreating dendrogram from sample...")

    linked = linkage(
        X_sample_scaled,
        method="ward",
        metric="euclidean"
    )

    plt.figure(
        figsize=(12, 6)
    )

    dendrogram(
        linked,
        truncate_mode="lastp",
        p=30,
        show_contracted=True
    )

    plt.title(
        f"Hierarchical Clustering Dendrogram "
        f"({sample_size:,} Row Sample)"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Distance"
    )

    dendrogram_path = os.path.join(
        output_directory,
        "dendrogram.png"
    )

    plt.savefig(
        dendrogram_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Dendrogram saved to:\n"
        f"{dendrogram_path}"
    )


    # =========================================================
    # 13B. HCA ON SAMPLE
    # =========================================================

    print("\nRunning HCA on sample...")

    hca = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        linkage="ward"
    )

    sample_labels = hca.fit_predict(
        X_sample_scaled
    )

    print(
        "Sample HCA completed."
    )


    # =========================================================
    # 13C. CALCULATE CLUSTER CENTROIDS
    # =========================================================

    print(
        "\nCalculating cluster centroids..."
    )

    centroids = []

    for cluster_id in range(N_CLUSTERS):

        cluster_points = X_sample_scaled[
            sample_labels == cluster_id
        ]

        if len(cluster_points) == 0:

            raise ValueError(
                f"Cluster {cluster_id} contains no sample points."
            )

        centroid = cluster_points.mean(
            axis=0
        )

        centroids.append(
            centroid
        )

    centroids = np.array(
        centroids
    )

    print(
        f"Centroid matrix shape: "
        f"{centroids.shape}"
    )


    # =========================================================
    # 13D. ASSIGN ALL DATA TO NEAREST CENTROID
    # =========================================================

    print(
        "\nAssigning all rides to the nearest cluster..."
    )

    # Calculate Euclidean distance from every row
    # to every cluster centroid.

    distances = np.linalg.norm(
        X_scaled[:, np.newaxis, :]
        - centroids[np.newaxis, :, :],
        axis=2
    )

    cluster_labels = np.argmin(
        distances,
        axis=1
    )

    df["cluster"] = cluster_labels

    print(
        "All rows successfully assigned to clusters."
    )


# =============================================================
# 14. CLUSTER DISTRIBUTION
# =============================================================

print("\n" + "=" * 70)
print("FINAL CLUSTER DISTRIBUTION")
print("=" * 70)

cluster_distribution = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print(
    cluster_distribution
)


# =============================================================
# 15. CLUSTER PERCENTAGES
# =============================================================

print("\nCluster percentages:")

cluster_percentages = (
    df["cluster"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

for cluster_id, percentage in cluster_percentages.items():

    count = cluster_distribution[
        cluster_id
    ]

    print(
        f"Cluster {cluster_id}: "
        f"{count:,} rides "
        f"({percentage:.2f}%)"
    )


# =============================================================
# 16. CLUSTER PROFILES
# =============================================================

print("\n" + "=" * 70)
print("CLUSTER PROFILES")
print("=" * 70)

profile = (
    df
    .groupby("cluster")[features]
    .mean()
    .round(2)
)

profile["ride_count"] = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

profile["percentage"] = (
    df["cluster"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(2)
)

print(
    profile
)


# =============================================================
# 17. SAVE CLUSTER PROFILE
# =============================================================

profile_file = os.path.join(
    output_directory,
    "cluster_profiles.csv"
)

profile.to_csv(
    profile_file
)

print(
    f"\nCluster profile saved to:\n"
    f"{profile_file}"
)


# =============================================================
# 18. SAVE CLUSTERED DATASET
# =============================================================

output_file = os.path.join(
    output_directory,
    "ncr_ride_bookings_clustered.csv"
)

df.to_csv(
    output_file,
    index=False
)


# =============================================================
# 19. FINAL SUMMARY
# =============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"\nInput dataset:"
    f"\n  {input_path}"
)

print(
    f"\nOutput dataset:"
    f"\n  {output_file}"
)

print(
    f"\nDendrogram:"
    f"\n  {dendrogram_path}"
)

print(
    f"\nCluster profile:"
    f"\n  {profile_file}"
)

print(
    f"\nNumber of rows:"
    f"\n  {len(df):,}"
)

print(
    f"\nNumber of clusters:"
    f"\n  {N_CLUSTERS}"
)

print(
    "\nFeatures used:"
)

for feature in features:
    print(
        f"  - {feature}"
    )

print(
    "\nHCA processing completed successfully."
)

