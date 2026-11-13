
from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
from pathlib import Path
import json
import uuid
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

REGISTRY_PATH = BASE_DIR / "data" / "metadata" / "datasets.json"

# Dataset upload configuration
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DATASETS_DIR = BASE_DIR / "data" / "datasets"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)




def load_dataset_registry():
    """Load and validate the dataset registry."""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        registry = json.load(file)

    if not isinstance(registry.get("datasets"), list):
        raise ValueError("Invalid dataset registry.")

    return registry


def get_dataset_path(dataset_id=None):
    """Resolve a registered dataset ID to a safe project file path."""
    registry = load_dataset_registry()

    selected_id = dataset_id or registry.get(
        "active_dataset", "online_retail"
    )

    dataset = next(
        (
            item for item in registry["datasets"]
            if item.get("id") == selected_id
        ),
        None
    )

    if dataset is None:
        raise ValueError("Selected dataset was not found.")

    dataset_path = (BASE_DIR / dataset["path"]).resolve()

    try:
        dataset_path.relative_to(BASE_DIR.resolve())
    except ValueError:
        raise ValueError("Dataset path is outside the project folder.")

    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset file does not exist.")

    return dataset_path






def inspect_uploaded_csv(file_path, mapping=None):
    """Read, normalize, and validate the complete uploaded CSV."""

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise ValueError("Could not read the CSV. Check its format and encoding.")

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    available_columns = list(df.columns)

    # Normalize column names for case-insensitive automatic detection.
    column_lookup = {}
    for column in available_columns:
        normalized_name = str(column).strip().lower()

        if normalized_name in column_lookup:
            raise ValueError(
                "Duplicate column names detected after normalization. "
                "Please provide a CSV with unique column names."
            )

        column_lookup[normalized_name] = column

    standard_aliases = {
        "customer_id": ["customerid", "customer id", "customer_id"],
        "invoice_id": [
            "invoiceno", "invoice no", "invoice number", "invoice_id"
        ],
        "invoice_date": [
            "invoicedate", "invoice date", "invoice_date"
        ],
        "revenue": ["revenue", "total revenue", "sales"],
        "quantity": ["quantity", "qty", "units"],
        "unit_price": ["unitprice", "unit price", "price", "priceeach"]
    }

    if mapping is None:
        mapping = {}
    elif not isinstance(mapping, dict):
        raise ValueError("Mapping must be a JSON object.")

    allowed_fields = set(standard_aliases)

    if any(field not in allowed_fields for field in mapping):
        raise ValueError("Mapping contains an unsupported field.")

    resolved_mapping = {}

    for field, column in mapping.items():
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"Invalid column mapping for '{field}'."
            )

        resolved_mapping[field] = column.strip()

    # Fill any unmapped fields using automatic column detection.
    for field, aliases in standard_aliases.items():
        if field not in resolved_mapping:
            matches = [
                column_lookup[alias]
                for alias in aliases
                if alias in column_lookup
            ]

            if len(matches) > 1:
                raise ValueError(
                    f"Multiple possible columns found for {field}. "
                    "Please provide a custom mapping."
                )

            if matches:
                resolved_mapping[field] = matches[0]

    required_fields = [
        "customer_id",
        "invoice_id",
        "invoice_date"
    ]

    missing = [
        field for field in required_fields
        if not resolved_mapping.get(field)
    ]

    if missing:
        raise ValueError(
            "Missing mappings for: " + ", ".join(missing)
        )

    for field, column in resolved_mapping.items():
        if column not in available_columns:
            raise ValueError(
                f"Mapped column '{column}' for '{field}' was not found."
            )

    revenue_column = resolved_mapping.get("revenue")
    quantity_column = resolved_mapping.get("quantity")
    unit_price_column = resolved_mapping.get("unit_price")

    if not revenue_column and not (
        quantity_column and unit_price_column
    ):
        raise ValueError(
            "Map Revenue, or map both Quantity and Unit Price."
        )

    normalized = pd.DataFrame()

    normalized["CustomerID"] = pd.to_numeric(
        df[resolved_mapping["customer_id"]],
        errors="coerce"
    )

    normalized["InvoiceNo"] = df[resolved_mapping["invoice_id"]]

    normalized["InvoiceDate"] = pd.to_datetime(
        df[resolved_mapping["invoice_date"]],
        errors="coerce"
    )

    if revenue_column:
        normalized["Revenue"] = pd.to_numeric(
            df[revenue_column],
            errors="coerce"
        )
    else:
        quantity = pd.to_numeric(
            df[quantity_column],
            errors="coerce"
        )
        unit_price = pd.to_numeric(
            df[unit_price_column],
            errors="coerce"
        )
        normalized["Revenue"] = quantity * unit_price

    normalized["InvoiceNo"] = normalized["InvoiceNo"].replace(
        r"^\\s*$", pd.NA, regex=True
    )

    normalized = normalized.dropna(
        subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Revenue"]
    )

    customer_count = normalized["CustomerID"].nunique()

    if customer_count < 4:
        raise ValueError(
            "At least four unique customers with valid data are required."
        )

    return {
        "columns": available_columns,
        "mapping": resolved_mapping,
        "row_count": int(len(normalized)),
        "customer_count": int(customer_count),
        "preview": normalized.head(10).fillna("").to_dict(
            orient="records"
        ),
        "normalized_data": normalized
    }


def get_customer_data(dataset_id=None):
    dataset_path = get_dataset_path(dataset_id)
    df = pd.read_csv(dataset_path)

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
    rfm = get_customer_data(request.args.get("dataset_id"))

    return jsonify({
        "total_customers": int(len(rfm)),
        "total_revenue": round(float(rfm["Monetary"].sum()), 2),
        "average_spending": round(float(rfm["Monetary"].mean()), 2),
        "average_frequency": round(float(rfm["Frequency"].mean()), 2)
    })



@app.route("/api/segments")
def segments():
    rfm = get_customer_data(request.args.get("dataset_id"))

    result = rfm.groupby("Cluster").agg(
        Segment=("Segment", "first"),
        Customers=("CustomerID", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Total_Revenue=("Monetary", "sum"),
        Average_Spending=("Monetary", "mean")
    ).reset_index()
    recommendations = {
    "High Value": "Offer VIP rewards and personalized benefits.",
    "Loyal Customers": "Reward repeat purchases and encourage referrals.",
    "Needs Attention": "Send reactivation offers to encourage another purchase.",
    "Regular Customers": "Use loyalty points and targeted offers to encourage repeat purchases."
}

    result["Recommendation"] = result["Segment"].map(recommendations)

    return jsonify({
        "segments": result.round(2).to_dict(orient="records")
    })



@app.route("/api/customers")
def customers():
    rfm = get_customer_data(request.args.get("dataset_id"))
    rfm = rfm.sort_values("Monetary", ascending=False)

    return jsonify({
        "customers": rfm.head(100).round(2).to_dict(orient="records")
    })





@app.route("/api/datasets", methods=["GET"])
def datasets():
    registry = load_dataset_registry()

    return jsonify({
        "datasets": [
            {
                "id": item["id"],
                "name": item["name"],
                "type": item.get("type", "uploaded"),
                "currency": item.get("currency", "GBP")
            }
            for item in registry["datasets"]
        ],
        "active_dataset": registry.get("active_dataset", "online_retail")
    })


@app.route("/api/datasets/activate", methods=["POST"])
def activate_dataset():
    data = request.get_json(silent=True) or {}
    dataset_id = data.get("dataset_id")

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return jsonify({"error": "A valid dataset_id is required."}), 400

    try:
        # Verify the dataset exists and its file is available.
        get_dataset_path(dataset_id)

        registry = load_dataset_registry()
        registry["active_dataset"] = dataset_id

        temporary_path = REGISTRY_PATH.with_suffix(".tmp")

        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=4)

        temporary_path.replace(REGISTRY_PATH)

        return jsonify({
            "message": "Dataset activated successfully.",
            "active_dataset": dataset_id
        })

    except (ValueError, FileNotFoundError):
        return jsonify({"error": "Dataset not found or unavailable."}), 404




@app.route("/api/datasets/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Please select a CSV file."}), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "No filename provided."}), 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    temporary_id = uuid.uuid4().hex
    temporary_path = UPLOAD_DIR / f"{temporary_id}.csv"

    try:
        uploaded_file.save(temporary_path)

        if temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)
            return jsonify({"error": "The uploaded file is empty."}), 400

        mapping_text = request.form.get("mapping")
        mapping = None

        if mapping_text:
            try:
                mapping = json.loads(mapping_text)
            except json.JSONDecodeError:
                raise ValueError("Mapping must contain valid JSON.")

            if not isinstance(mapping, dict):
                raise ValueError("Mapping must be a JSON object.")

        result = inspect_uploaded_csv(temporary_path, mapping)

        return jsonify({
            "upload_id": temporary_id,
            "filename": uploaded_file.filename,
            "mapping": result["mapping"],
            "columns": result["columns"],
            "row_count": result["row_count"],
            "customer_count": result["customer_count"],
            "preview": result["preview"][:10]
        }), 200

    except ValueError as error:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 400

    except Exception:
        temporary_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Could not process the uploaded CSV."
        }), 400


@app.route("/api/datasets/save", methods=["POST"])
def save_dataset():
    data = request.get_json(silent=True) or {}

    upload_id = data.get("upload_id")
    dataset_name = data.get("name")
    currency = data.get("currency", "GBP")
    mapping = data.get("mapping")

    if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        return jsonify({"error": "Invalid upload ID."}), 400

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        return jsonify({"error": "Please provide a dataset name."}), 400

    dataset_name = dataset_name.strip()

    if len(dataset_name) > 80:
        return jsonify({"error": "Dataset name must be 80 characters or fewer."}), 400

    if currency not in ["GBP", "INR", "USD", "EUR"]:
        return jsonify({"error": "Unsupported currency."}), 400

    temporary_path = UPLOAD_DIR / f"{upload_id}.csv"

    if not temporary_path.is_file():
        return jsonify({"error": "Uploaded file not found. Please upload again."}), 404

    dataset_id = uuid.uuid4().hex
    final_path = DATASETS_DIR / f"{dataset_id}.csv"
    staging_path = DATASETS_DIR / f"{dataset_id}.tmp"

    try:
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("Mapping must be a JSON object.")

        result = inspect_uploaded_csv(temporary_path, mapping)
        normalized = result["normalized_data"]

        normalized.to_csv(staging_path, index=False)
        staging_path.replace(final_path)

        registry = load_dataset_registry()

        registry["datasets"].append({
            "id": dataset_id,
            "name": dataset_name,
            "path": f"data/datasets/{dataset_id}.csv",
            "type": "uploaded",
            "currency": currency,
            "mapping": result["mapping"],
            "row_count": result["row_count"],
            "customer_count": result["customer_count"]
        })

        registry_temp = REGISTRY_PATH.with_suffix(".tmp")

        with open(registry_temp, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=4)

        registry_temp.replace(REGISTRY_PATH)
        temporary_path.unlink(missing_ok=True)

        return jsonify({
            "message": "Dataset saved successfully.",
            "dataset_id": dataset_id,
            "name": dataset_name,
            "row_count": result["row_count"],
            "customer_count": result["customer_count"],
            "currency": currency
        }), 201

    except ValueError as error:
        staging_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 400

    except Exception:
        staging_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not save the dataset."}), 500



if __name__ == "__main__":
    app.run(debug=True)
