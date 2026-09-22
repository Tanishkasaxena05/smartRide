import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartRide - HCA Clustering",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 SmartRide - Hierarchical Cluster Analysis")
st.markdown(
    """
    This application performs **Hierarchical Cluster Analysis (HCA)**
    on the SmartRide booking dataset.
    """
)


# ============================================================
# LOAD DATASET AUTOMATICALLY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(
    BASE_DIR,
    "data",
    "ncr_ride_bookings.csv"
)

if not os.path.exists(input_path):
    st.error(
        f"""
        ❌ Dataset not found.

        Expected file:

        `{input_path}`

        Please make sure your project structure is:

        Project 1/
        ├── app.py
        └── data/
            └── ncr_ride_bookings.csv
        """
    )
    st.stop()


try:
    df = pd.read_csv(input_path)
except Exception as e:
    st.error(f"Error reading dataset: {e}")
    st.stop()


# Clean column names
df.columns = df.columns.str.strip()


st.success(
    f"✅ Dataset loaded automatically: `{input_path}`"
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ HCA Settings")

n_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)

sample_size = st.sidebar.slider(
    "Sample size for large datasets",
    min_value=500,
    max_value=10000,
    value=5000,
    step=500
)

direct_hca_limit = st.sidebar.slider(
    "Direct HCA row limit",
    min_value=1000,
    max_value=20000,
    value=15000,
    step=1000
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

with col3:
    st.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

with col4:
    st.metric(
        "Memory Usage",
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )


# ============================================================
# SHOW DATA
# ============================================================

with st.expander("👀 View Dataset Preview"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# REQUIRED FEATURES
# ============================================================

st.header("🎯 Clustering Features")

st.write(
    "The following numerical features will be used for HCA:"
)

features = [
    "Booking Value",
    "Ride Distance",
    "Driver Ratings",
    "Customer Rating",
    "Avg VTAT",
    "Avg CTAT"
]


# Check which features exist
missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]


if missing_features:

    st.error(
        "The following required columns are missing from the dataset:"
    )

    for feature in missing_features:
        st.write(f"- `{feature}`")

    st.write("Available columns:")

    st.write(list(df.columns))

    st.stop()


st.success("✅ All required clustering features are available.")


# ============================================================
# FEATURE DATA
# ============================================================

X = df[features].copy()


# Convert everything to numeric
for column in features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# MISSING VALUE INFORMATION
# ============================================================

st.subheader("Missing Values in Clustering Features")

missing_table = pd.DataFrame({
    "Feature": features,
    "Missing Values": [
        X[column].isna().sum()
        for column in features
    ],
    "Missing Percentage": [
        round(
            X[column].isna().mean() * 100,
            2
        )
        for column in features
    ]
})

st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA PREPARATION
# ============================================================

st.header("🔧 Data Preparation")

# Remove columns that contain no numeric values at all
valid_features = [
    column
    for column in features
    if X[column].notna().sum() > 0
]


if len(valid_features) == 0:

    st.error(
        "❌ None of the selected features contain valid numeric data."
    )

    st.stop()


if len(valid_features) < len(features):

    removed_features = [
        column
        for column in features
        if column not in valid_features
    ]

    st.warning(
        "The following features were removed because they contain "
        "no valid numeric values:"
    )

    for feature in removed_features:
        st.write(f"- `{feature}`")


X = X[valid_features]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)


st.success(
    f"""
    ✅ Data preparation completed.

    Features used: {len(valid_features)}

    Rows used: {len(X_scaled):,}
    """
)


# ============================================================
# RUN HCA
# ============================================================

st.header("🔬 Hierarchical Cluster Analysis")

st.write(
    f"""
    **Number of clusters:** {n_clusters}

    **Dataset size:** {len(X_scaled):,} rows
    """
)


run_hca = st.button(
    "🚀 Run HCA",
    type="primary",
    use_container_width=True
)


if run_hca:

    # --------------------------------------------------------
    # CHECK CLUSTER COUNT
    # --------------------------------------------------------

    if n_clusters >= len(X_scaled):

        st.error(
            "Number of clusters must be smaller than the number of rows."
        )

        st.stop()


    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    progress = st.progress(0)

    status = st.empty()


    # ========================================================
    # DIRECT HCA
    # ========================================================

    if len(X_scaled) <= direct_hca_limit:

        status.info(
            "Running direct Hierarchical Agglomerative Clustering..."
        )

        progress.progress(25)


        # ----------------------------------------------------
        # DENDROGRAM
        # ----------------------------------------------------

        st.subheader("🌳 Hierarchical Dendrogram")

        # Create linkage matrix
        Z = linkage(
            X_scaled,
            method="ward"
        )

        progress.progress(50)


        fig, ax = plt.subplots(
            figsize=(14, 7)
        )

        dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax
        )

        ax.set_title(
            "Hierarchical Clustering Dendrogram"
        )

        ax.set_xlabel(
            "Cluster / Sample"
        )

        ax.set_ylabel(
            "Distance"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------------
        # AGGLOMERATIVE CLUSTERING
        # ----------------------------------------------------

        status.info(
            "Creating clusters..."
        )

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

        labels = model.fit_predict(
            X_scaled
        )

        progress.progress(75)


        method_used = "Direct HCA"


    # ========================================================
    # HYBRID HCA FOR LARGE DATASETS
    # ========================================================

    else:

        status.info(
            f"""
            Dataset is large ({len(X_scaled):,} rows).

            Using hybrid HCA with a sample of
            {min(sample_size, len(X_scaled)):,} rows...
            """
        )

        progress.progress(15)


        # ----------------------------------------------------
        # RANDOM SAMPLE
        # ----------------------------------------------------

        rng = np.random.default_rng(
            42
        )

        actual_sample_size = min(
            sample_size,
            len(X_scaled)
        )

        sample_indices = rng.choice(
            len(X_scaled),
            size=actual_sample_size,
            replace=False
        )

        X_sample = X_scaled[
            sample_indices
        ]


        progress.progress(30)


        # ----------------------------------------------------
        # LINKAGE ON SAMPLE
        # ----------------------------------------------------

        status.info(
            "Building hierarchical structure from sample..."
        )

        Z = linkage(
            X_sample,
            method="ward"
        )

        progress.progress(45)


        # ----------------------------------------------------
        # DENDROGRAM
        # ----------------------------------------------------

        st.subheader("🌳 Hierarchical Dendrogram")

        fig, ax = plt.subplots(
            figsize=(14, 7)
        )

        dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax
        )

        ax.set_title(
            "Hierarchical Clustering Dendrogram "
            "(Sampled Data)"
        )

        ax.set_xlabel(
            "Cluster / Sample"
        )

        ax.set_ylabel(
            "Distance"
        )

        st.pyplot(fig)

        plt.close(fig)


        progress.progress(60)


        # ----------------------------------------------------
        # CLUSTER SAMPLE
        # ----------------------------------------------------

        status.info(
            "Creating sample clusters..."
        )

        sample_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

        sample_labels = sample_model.fit_predict(
            X_sample
        )


        progress.progress(70)


        # ----------------------------------------------------
        # CALCULATE CLUSTER CENTROIDS
        # ----------------------------------------------------

        centroids = []

        for cluster_id in range(
            n_clusters
        ):

            cluster_points = X_sample[
                sample_labels == cluster_id
            ]

            if len(cluster_points) == 0:

                # Fallback in case a cluster is empty
                centroid = np.mean(
                    X_sample,
                    axis=0
                )

            else:

                centroid = np.mean(
                    cluster_points,
                    axis=0
                )

            centroids.append(
                centroid
            )


        centroids = np.array(
            centroids
        )


        # ----------------------------------------------------
        # ASSIGN ALL DATA TO NEAREST CENTROID
        # ----------------------------------------------------

        status.info(
            "Assigning all rides to the nearest cluster..."
        )


        labels = np.empty(
            len(X_scaled),
            dtype=int
        )


        # Process in batches to avoid excessive memory usage
        batch_size = 10000

        for start in range(
            0,
            len(X_scaled),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X_scaled)
            )

            batch = X_scaled[
                start:end
            ]

            distances = np.linalg.norm(
                batch[:, np.newaxis, :] -
                centroids[np.newaxis, :, :],
                axis=2
            )

            labels[start:end] = np.argmin(
                distances,
                axis=1
            )


        progress.progress(85)

        method_used = "Hybrid HCA"


    # ========================================================
    # ADD CLUSTER LABELS
    # ========================================================

    result_df = df.copy()

    result_df["Cluster"] = labels


    # ========================================================
    # CLUSTER DISTRIBUTION
    # ========================================================

    st.header("📈 Cluster Distribution")

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_distribution = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Number of Rides": cluster_counts.values
    })


    col1, col2 = st.columns(2)


    with col1:

        st.dataframe(
            cluster_distribution,
            use_container_width=True,
            hide_index=True
        )


    with col2:

        chart_data = cluster_distribution.set_index(
            "Cluster"
        )

        st.bar_chart(
            chart_data
        )


    # ========================================================
    # CLUSTER PROFILE
    # ========================================================

    st.header("📊 Cluster Profiles")


    profile_columns = [
        column
        for column in valid_features
        if column in result_df.columns
    ]


    # Make sure profile columns are numeric
    for column in profile_columns:

        result_df[column] = pd.to_numeric(
            result_df[column],
            errors="coerce"
        )


    cluster_profile = (
        result_df
        .groupby("Cluster")[profile_columns]
        .mean()
        .round(2)
    )


    # Add number of rides
    cluster_profile.insert(
        0,
        "Number of Rides",
        result_df["Cluster"]
        .value_counts()
        .sort_index()
    )


    st.dataframe(
        cluster_profile,
        use_container_width=True
    )


    # ========================================================
    # DETAILED CLUSTER INFORMATION
    # ========================================================

    st.header("🔍 Cluster Details")


    for cluster_id in sorted(
        result_df["Cluster"].unique()
    ):

        cluster_data = result_df[
            result_df["Cluster"] == cluster_id
        ]


        with st.expander(
            f"Cluster {cluster_id} - "
            f"{len(cluster_data):,} rides"
        ):

            detail_columns = [
                column
                for column in valid_features
                if column in cluster_data.columns
            ]


            detail_profile = (
                cluster_data[
                    detail_columns
                ]
                .mean()
                .round(2)
                .to_frame(
                    "Average Value"
                )
            )


            st.dataframe(
                detail_profile,
                use_container_width=True
            )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.header("💾 Download Results")


    csv_data = result_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(
        label="⬇️ Download Clustered Dataset",
        data=csv_data,
        file_name="ncr_ride_bookings_clustered.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # SAVE RESULTS LOCALLY
    # ========================================================

    output_dir = os.path.join(
        BASE_DIR,
        "data",
        "cleaned"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    output_path = os.path.join(
        output_dir,
        "ncr_ride_bookings_clustered.csv"
    )


    result_df.to_csv(
        output_path,
        index=False
    )


    # Save cluster profile
    profile_path = os.path.join(
        output_dir,
        "cluster_profiles.csv"
    )


    cluster_profile.to_csv(
        profile_path
    )


    # ========================================================
    # COMPLETION
    # ========================================================

    progress.progress(100)

    status.success(
        f"""
        ✅ HCA completed successfully!

        Method used: **{method_used}**

        Number of clusters: **{n_clusters}**

        Results saved to:
        `{output_path}`
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SmartRide | Hierarchical Cluster Analysis"
)