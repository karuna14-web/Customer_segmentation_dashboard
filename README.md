# Customer Segmentation Dashboard Using K-Means Clustering

An interactive web dashboard that analyzes customer purchasing behavior using RFM analysis and K-Means clustering. It helps businesses understand customer groups and make data-driven marketing decisions.

## Features

- User registration, login and logout
- Secure dataset access and ownership
- CSV dataset upload and dataset selection
- Customer segmentation using RFM analysis and K-Means clustering
- Customer retention risk analysis
- Elbow Method visualization
- Silhouette Score comparison
- RFM segment profiles and interpretations
- Interactive charts and customer tables
- Customer search and segment filtering
- Segment-based business recommendations

## Technology Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Plotly.js
- HTML, CSS and JavaScript

## Machine Learning Workflow

1. Data ingestion
2. Data preprocessing
3. RFM feature calculation
4. Feature scaling
5. K-Means clustering
6. Cluster profiling and segment interpretation
7. Dashboard visualization

## RFM Analysis

- **Recency:** How recently a customer made a purchase.
- **Frequency:** How often a customer purchased.
- **Monetary:** How much a customer spent.

## Clustering Evaluation

- **Elbow Method:** Compares inertia across different values of K.
- **Silhouette Score:** Evaluates clustering separation and cohesion.

The dashboard currently uses four business-oriented customer segments. The evaluation metrics are presented to support the analysis and explain the selection of clusters.

## Customer Segments

- High Value
- Loyal Customers
- Needs Attention
- Regular Customers

Segment interpretations are based on customer RFM profiles.

## Project Structure

    customer_segmentation_recovered/
    ├── backend/
    ├── frontend/
    ├── notebooks/
    ├── data/
    ├── models/
    ├── docs/
    ├── backups/
    └── requirements.txt

## Running the Project

### 1. Clone the repository

    git clone https://github.com/karuna14-web/Customer_segmentation_dashboard.git
    cd Customer_segmentation_dashboard

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Configure environment variables

Set the required environment variables, including FLASK_SECRET_KEY, before starting the application.

### 4. Run the Flask application

    python backend/app.py

### 5. Open the application

Open the local Flask URL shown in the terminal in your browser.

## Dataset

The project supports uploading CSV datasets through the dashboard.

## Project Purpose

This project demonstrates the use of machine learning, data analysis, data visualization, web development and software engineering to create a customer analytics application.

---

Developed as a college project.
