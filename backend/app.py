
from flask import Flask, jsonify, send_from_directory
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
FRONTEND_DIR = BASE_DIR / "frontend"


def get_customer_data():
    df = pd.read_csv(DATA_PATH)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )
    df["Revenue"] = pd.to_numeric(
        df["Revenue"], errors="coerce"
    )

    df = df.dropna(
        subset=["CustomerID", "InvoiceDate", "Revenue", "InvoiceNo"]
    )

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x:
                 (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    rfm = rfm[
        rfm[["Recency", "Frequency", "Monetary"]]
        .notna().all(axis=1)
    ]

    if len(rfm) < 4:
        raise ValueError("At least four customers are required.")

    scaler = StandardScaler()
    features = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = model.fit_predict(features)
    
    # Create meaningful names for the four clusters
    stats = rfm.groupby("Cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean")
    )

    remaining = list(stats.index)
    labels = {}

    # Highest average spending
    high_value = stats.loc[remaining, "Monetary"].idxmax()
    labels[high_value] = "High Value"
    remaining.remove(high_value)

    # Highest purchase frequency among remaining clusters
    loyal = stats.loc[remaining, "Frequency"].idxmax()
    labels[loyal] = "Loyal Customers"
    remaining.remove(loyal)

    # Highest recency means customers have been inactive longer
    attention = stats.loc[remaining, "Recency"].idxmax()
    labels[attention] = "Needs Attention"
    remaining.remove(attention)

    # Last cluster
    labels[remaining[0]] = "Regular Customers"

    rfm["Segment"] = rfm["Cluster"].map(labels)

    return rfm


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/summary")
def summary():
    rfm = get_customer_data()

    return jsonify({
        "total_customers": int(len(rfm)),
        "total_revenue": round(float(rfm["Monetary"].sum()), 2),
        "average_spending": round(float(rfm["Monetary"].mean()), 2),
        "average_frequency": round(float(rfm["Frequency"].mean()), 2)
    })



@app.route("/api/segments")
def segments():
    rfm = get_customer_data()

    result = rfm.groupby("Cluster").agg(
        Segment=("Segment", "first"),
        Customers=("CustomerID", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Total_Revenue=("Monetary", "sum"),
        Average_Spending=("Monetary", "mean")
    ).reset_index()

    return jsonify({
        "segments": result.round(2).to_dict(orient="records")
    })



@app.route("/api/customers")
def customers():
    rfm = get_customer_data()
    rfm = rfm.sort_values("Monetary", ascending=False)

    return jsonify({
        "customers": rfm.head(100).round(2).to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)