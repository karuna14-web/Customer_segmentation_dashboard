
from flask import Flask, jsonify, send_from_directory, request, session, redirect
from authlib.integrations.flask_client import OAuth
import os
import auth_db
import pandas as pd
from pathlib import Path
from functools import wraps
import json
import uuid
import re
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)
app.config["SERVER_NAME"] = "segmentiq-p479.onrender.com"

app.config["SECRET_KEY"] = os.environ["FLASK_SECRET_KEY"]
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Google OAuth configuration
oauth = OAuth(app)

oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)


auth_db.init_auth_db()


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
    """Resolve a dataset only when the logged-in user can access it."""

    user_id = session.get("user_id")

    if not user_id:
        raise ValueError("Authentication required.")

    registry = load_dataset_registry()

    # Use the current user's active dataset when no ID is supplied.
    selected_id = dataset_id or auth_db.get_active_dataset(user_id)

    dataset = next(
        (
            item for item in registry["datasets"]
            if item.get("id") == selected_id
        ),
        None
    )

    if dataset is None:
        raise ValueError("Selected dataset was not found.")

    # Online Retail is the only shared dataset.
    is_shared_dataset = (
        dataset.get("id") == "online_retail"
        and dataset.get("owner_id") is None
    )

    is_owned_dataset = dataset.get("owner_id") == user_id

    if not is_shared_dataset and not is_owned_dataset:
        raise ValueError("You do not have access to this dataset.")

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



def evaluate_clustering(features):
    """Compare K values using inertia and silhouette score."""

    results = []
    total_customers = len(features)

    max_k = min(8, total_customers - 1)

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        cluster_labels = model.fit_predict(features)

        score = silhouette_score(
            features,
            cluster_labels,
            sample_size=min(1000, total_customers),
            random_state=42
        )

        results.append({
            "k": int(k),
            "inertia": round(float(model.inertia_), 2),
            "silhouette_score": round(float(score), 4)
        })

    best_silhouette_k = max(
        results,
        key=lambda item: item["silhouette_score"]
    )["k"]

    return {
        "selected_k": 4,
        "best_silhouette_k": int(best_silhouette_k),
        "silhouette_sample_size": min(1000, total_customers),
        "evaluations": results
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




def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        user_id = session.get("user_id")

        if not user_id or not auth_db.get_user_by_id(user_id):
            session.clear()
            return jsonify({"error": "Authentication required."}), 401

        return view_function(*args, **kwargs)

    return wrapped_view


@app.route("/")
def home():
    user_id = session.get("user_id")

    if user_id and auth_db.get_user_by_id(user_id):
        return send_from_directory(FRONTEND_DIR, "index.html")

    return send_from_directory(FRONTEND_DIR, "home.html")


@app.route("/login")
def login_page():
    user_id = session.get("user_id")

    if user_id and auth_db.get_user_by_id(user_id):
        return redirect("/")

    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/register")
def register_page():
    user_id = session.get("user_id")

    if user_id and auth_db.get_user_by_id(user_id):
        return redirect("/")

    return send_from_directory(FRONTEND_DIR, "register.html")




@app.route("/api/exchange-rate")
def exchange_rate():
    base = request.args.get("base", "").upper().strip()
    target = request.args.get("target", "").upper().strip()

    allowed_currencies = {"GBP", "INR", "USD", "EUR"}

    if base not in allowed_currencies or target not in allowed_currencies:
        return jsonify({"error": "Unsupported currency."}), 400

    if base == target:
        return jsonify({
            "base": base,
            "target": target,
            "rate": 1,
            "date": None
        })

    url = "https:" + "/" + "/" + "api.frankfurter.dev" + "/v2/rate/" + base + "/" + target

    try:
        with urlopen(
            Request(
                url,
                headers={"User-Agent": "CustomerSegmentationDashboard/1.0"}
            ),
            timeout=10
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

        return jsonify({
            "base": result["base"],
            "target": result["quote"],
            "rate": result["rate"],
            "date": result["date"]
        })

    except Exception:
        return jsonify({
            "error": "Exchange rate service is temporarily unavailable."
        }), 503

def handle_dataset_errors(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        try:
            return view_function(*args, **kwargs)
        except ValueError as error:
            return jsonify({"error": str(error)}), 403
        except FileNotFoundError:
            return jsonify({"error": "Dataset file not found."}), 404
    return wrapped_view


@app.route("/api/summary")
@login_required
def summary():
    try:
        rfm = get_customer_data(request.args.get("dataset_id"))

        return jsonify({
            "total_customers": int(len(rfm)),
            "total_revenue": round(float(rfm["Monetary"].sum()), 2),
            "average_spending": round(float(rfm["Monetary"].mean()), 2),
            "average_frequency": round(float(rfm["Frequency"].mean()), 2)
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 403

    except FileNotFoundError:
        return jsonify({"error": "Dataset file not found."}), 404



@app.route("/api/segments")
@login_required
@handle_dataset_errors
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



@app.route("/api/ml-analysis")
@login_required
@handle_dataset_errors
def ml_analysis():
    rfm = get_customer_data(request.args.get("dataset_id"))

    features = StandardScaler().fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    evaluation = evaluate_clustering(features)

    profiles = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean")
    ).reset_index()

    # Rank segment averages to create distinct RFM interpretations.
    profiles["Recency_Rank"] = profiles["Average_Recency"].rank(
        ascending=True, pct=True
    )
    profiles["Frequency_Rank"] = profiles["Average_Frequency"].rank(
        ascending=True, pct=True
    )
    profiles["Monetary_Rank"] = profiles["Average_Monetary"].rank(
        ascending=True, pct=True
    )

    def interpret_profile(row):
        recency_text = (
            "Recent purchases"
            if row["Recency_Rank"] <= 0.25
            else "Longer time since purchase"
            if row["Recency_Rank"] >= 0.75
            else "Moderate purchase recency"
        )

        frequency_text = (
            "High purchase frequency"
            if row["Frequency_Rank"] >= 0.75
            else "Low purchase frequency"
            if row["Frequency_Rank"] <= 0.25
            else "Moderate purchase frequency"
        )

        monetary_text = (
            "High spending"
            if row["Monetary_Rank"] >= 0.75
            else "Low spending"
            if row["Monetary_Rank"] <= 0.25
            else "Moderate spending"
        )

        return (
            f"{recency_text} | "
            f"{frequency_text} | "
            f"{monetary_text}"
        )

    profiles["RFM_Interpretation"] = profiles.apply(
        interpret_profile,
        axis=1
    )

    profiles = profiles.drop(
        columns=["Recency_Rank", "Frequency_Rank", "Monetary_Rank"]
    )

    return jsonify({
        "clustering": evaluation,
        "segment_profiles": profiles.round(2).to_dict(
            orient="records"
        )
    })



@app.route("/api/customers")
@login_required
@handle_dataset_errors
def customers():
    rfm = get_customer_data(request.args.get("dataset_id"))
    rfm = rfm.sort_values("Monetary", ascending=False)

    return jsonify({
        "customers": rfm.head(100).round(2).to_dict(orient="records")
    })






@app.route("/api/retention-risk")
@login_required
@handle_dataset_errors
def retention_risk():
    dataset_id = request.args.get("dataset_id")
    rfm = get_customer_data(dataset_id).copy()

    # Calculate relative percentile ranks
    recency_rank = rfm["Recency"].rank(pct=True)
    frequency_rank = rfm["Frequency"].rank(pct=True)
    monetary_rank = rfm["Monetary"].rank(pct=True)

    # Higher score means greater retention risk
    rfm["Risk_Score"] = (
        recency_rank * 50
        + (1 - frequency_rank) * 25
        + (1 - monetary_rank) * 25
    )

    # Classify customers
    def classify_risk(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    rfm["Risk_Level"] = rfm["Risk_Score"].apply(classify_risk)

    # Explain the main risk factors
    def get_risk_factors(row):
        factors = []

        if row["Recency_Rank"] >= 0.70:
            factors.append("Long time since last purchase")

        if row["Frequency_Rank"] <= 0.30:
            factors.append("Low purchase frequency")

        if row["Monetary_Rank"] <= 0.30:
            factors.append("Low total spending")

        if not factors:
            factors.append("No major relative risk factor")

        return factors

    rfm["Recency_Rank"] = recency_rank
    rfm["Frequency_Rank"] = frequency_rank
    rfm["Monetary_Rank"] = monetary_rank

    rfm["Risk_Factors"] = rfm.apply(get_risk_factors, axis=1)

    # Recommend retention actions
    def get_retention_action(row):
        if row["Risk_Level"] == "High":
            if row["Segment"] == "High Value":
                return "Provide VIP attention and a personalized retention offer."
            return "Send a targeted reactivation offer."

        elif row["Risk_Level"] == "Medium":
            return "Send a personalized reminder or loyalty reward."

        return "Maintain engagement with loyalty benefits."

    rfm["Retention_Action"] = rfm.apply(
        get_retention_action, axis=1
    )

    rfm = rfm.sort_values("Risk_Score", ascending=False)

    risk_counts = (
        rfm["Risk_Level"]
        .value_counts()
        .to_dict()
    )

    customers = rfm[
        [
            "CustomerID",
            "Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "Risk_Score",
            "Risk_Level",
            "Risk_Factors",
            "Retention_Action"
        ]
    ].copy()

    customers["Risk_Score"] = customers["Risk_Score"].round(2)
    customers["Monetary"] = customers["Monetary"].round(2)

    return jsonify({
        "total_customers": int(len(rfm)),
        "high_risk": int(risk_counts.get("High", 0)),
        "medium_risk": int(risk_counts.get("Medium", 0)),
        "low_risk": int(risk_counts.get("Low", 0)),
        "customers": customers.to_dict(orient="records")
    })



@app.route("/api/datasets", methods=["GET"])
@login_required
def datasets():
    user_id = session["user_id"]
    registry = load_dataset_registry()

    # Show only this user's datasets and the shared Online Retail dataset.
    accessible = [
        item for item in registry["datasets"]
        if (
            item.get("owner_id") == user_id
            or (
                item.get("id") == "online_retail"
                and item.get("owner_id") is None
            )
        )
    ]

    accessible_ids = {item["id"] for item in accessible}
    active_dataset = auth_db.get_active_dataset(user_id)

    # Recover safely if the saved active dataset is no longer accessible.
    if active_dataset not in accessible_ids:
        active_dataset = "online_retail"
        auth_db.set_active_dataset(user_id, active_dataset)

    return jsonify({
        "datasets": [
            {
                "id": item["id"],
                "name": item["name"],
                "type": item.get("type", "uploaded"),
                "currency": item.get("currency", "GBP")
            }
            for item in accessible
        ],
        "active_dataset": active_dataset
    })

@app.route("/api/datasets/activate", methods=["POST"])
@login_required
def activate_dataset():
    data = request.get_json(silent=True) or {}
    dataset_id = data.get("dataset_id")

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return jsonify({"error": "A valid dataset_id is required."}), 400

    try:
        get_dataset_path(dataset_id)

    except FileNotFoundError:
        return jsonify({"error": "Dataset not found."}), 404

    except ValueError as error:
        if str(error) == "You do not have access to this dataset.":
            return jsonify({"error": str(error)}), 403

        return jsonify({"error": str(error)}), 404

    auth_db.set_active_dataset(session["user_id"], dataset_id)

    return jsonify({
        "message": "Dataset activated successfully.",
        "active_dataset": dataset_id
    })

@app.route("/api/datasets/upload", methods=["POST"])
@login_required
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
    metadata_path = UPLOAD_DIR / f"{temporary_id}.json"

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

        # Associate this temporary upload with the logged-in user.
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump({"owner_id": session["user_id"]}, file)

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
        metadata_path.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 400

    except Exception:
        temporary_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return jsonify({
            "error": "Could not process the uploaded CSV."
        }), 400


@app.route("/api/datasets/save", methods=["POST"])
@login_required
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
    metadata_path = UPLOAD_DIR / f"{upload_id}.json"

    if not temporary_path.is_file() or not metadata_path.is_file():
        return jsonify({
            "error": "Uploaded file not found or expired. Please upload again."
        }), 404

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            upload_metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return jsonify({
            "error": "Upload ownership could not be verified. Please upload again."
        }), 403

    if upload_metadata.get("owner_id") != session["user_id"]:
        return jsonify({
            "error": "You do not have permission to save this upload."
        }), 403

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
            "owner_id": session["user_id"],
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
        metadata_path.unlink(missing_ok=True)

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




# ================= AUTHENTICATION ROUTES =================

@app.route("/api/auth/register", methods=["POST"])
def register():
    if auth_db.get_setting("registration_open", "0") != "1":
        return jsonify({"error": "Registration is currently closed."}), 403

    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not name or len(name) > 80:
        return jsonify({"error": "Enter a valid name (maximum 80 characters)."}), 400

    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
        return jsonify({"error": "Enter a valid email address."}), 400

    if not isinstance(password, str) or len(password) < 12 or len(password) > 128:
        return jsonify({"error": "Password must contain 12 to 128 characters."}), 400

    try:
        user_id = auth_db.create_user(name, email, password, role="user")
        return jsonify({"message": "Account created successfully.", "user_id": user_id}), 201
    except Exception:
        return jsonify({"error": "An account with this email may already exist."}), 409


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not email or not isinstance(password, str):
        return jsonify({"error": "Email and password are required."}), 400

    user = auth_db.get_user_by_email(email)

    if not user or not auth_db.verify_password(user, password):
        return jsonify({"error": "Invalid email or password."}), 401

    session.clear()
    session["user_id"] = user["id"]
    session["role"] = user["role"]

    return jsonify({
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }), 200


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logout successful."}), 200


@app.route("/api/auth/me", methods=["GET"])
def current_user():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"authenticated": False}), 401

    user = auth_db.get_user_by_id(user_id)

    if not user:
        session.clear()
        return jsonify({"authenticated": False}), 401

    return jsonify({
        "authenticated": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }), 200



# ================= GOOGLE AUTHENTICATION =================

@app.route("/auth/google")
def google_login():
    base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    print("GOOGLE BASE URL:", repr(base_url))

    return oauth.google.authorize_redirect(
        base_url + "/auth/google/callback"
    )

@app.route("/auth/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo")

        if not user_info:
            user_info = oauth.google.userinfo(token=token)

        email = str(user_info.get("email", "")).strip().lower()
        name = str(user_info.get("name", "")).strip()
        email_verified = user_info.get("email_verified", False)

        print("GOOGLE EMAIL EXISTS:", bool(email))
        print("GOOGLE EMAIL VERIFIED:", repr(email_verified))

        if not email or not email_verified:
            return redirect("/login")

        user = auth_db.get_user_by_email(email)
        print("GOOGLE USER FOUND:", bool(user))

        if not user:
            registration_open = auth_db.get_setting("registration_open", "0")
            print("GOOGLE REGISTRATION OPEN:", repr(registration_open))

            if registration_open != "1":
                return redirect("/login")

            random_password = uuid.uuid4().hex + uuid.uuid4().hex

            user_id = auth_db.create_user(
                name or email.split("@")[0],
                email,
                random_password,
                role="user"
            )

            user = auth_db.get_user_by_id(user_id)

        session.clear()
        session["user_id"] = user["id"]
        session["role"] = user["role"]

        return redirect("/")

    except Exception as e:
        print("GOOGLE OAUTH ERROR:", repr(e))
        return redirect("/login")


# ================= END GOOGLE AUTHENTICATION =================


# ================= END AUTHENTICATION ROUTES =================


if __name__ == "__main__":
    app.run(debug=True)
