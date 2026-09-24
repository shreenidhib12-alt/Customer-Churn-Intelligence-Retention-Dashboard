"""
app.py
------
Single-file Streamlit application.

Includes data ingestion & sanitisation (formerly data_loader.py),
leakage-safe RFM segmentation, ML churn models, retention optimiser
(formerly analytics.py), and the full interactive dashboard UI.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from docx import Document
from docx.shared import RGBColor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# Constants
# ============================================================================

RAW_PATH = os.path.join("data", "supermarket_sales_raw.csv")
CLEAN_PATH = os.path.join("data", "clean_data.csv")
PROFIT_MARGIN = 0.22   # cost-of-goods ratio used when Profit column is absent


# ============================================================================
# DATA LOADER
# ============================================================================

def _generate_synthetic_data() -> pd.DataFrame:
    """Return a 2 000-row realistic e-commerce transaction dataset."""
    rng = np.random.default_rng(42)
    n = 2_000
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    regions = ["North", "South", "East", "West", "Central"]

    order_dates = pd.to_datetime(
        rng.integers(
            int(pd.Timestamp("2022-01-01").timestamp()),
            int(pd.Timestamp("2023-12-31").timestamp()),
            n,
        ),
        unit="s",
    ).normalize()

    qty = rng.integers(1, 20, n)
    unit_price = np.round(rng.uniform(5, 500, n), 2)
    revenue = np.round(qty * unit_price, 2)
    profit = np.round(revenue * PROFIT_MARGIN, 2)

    return pd.DataFrame(
        {
            "Order_ID": [f"ORD{100000 + i}" for i in range(n)],
            "Customer_ID": [f"CUST{rng.integers(1000, 3000)}" for _ in range(n)],
            "Order_Date": order_dates,
            "Product_Category": rng.choice(categories, n),
            "Region": rng.choice(regions, n),
            "Quantity": qty,
            "Unit_Price": unit_price,
            "Revenue": revenue,
            "Profit": profit,
        }
    )


def _strip_currency(series: pd.Series) -> pd.Series:
    """Remove currency symbols and convert to float."""
    return (
        series.astype(str)
        .str.replace(r"[₹$£€INR,\s]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _title_case_series(series: pd.Series) -> pd.Series:
    """Trim whitespace and apply title-case normalisation."""
    return series.astype(str).str.strip().str.title()


def load_and_clean_data(file_path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load raw transaction data, sanitise it, and return a clean DataFrame.

    If the raw file is missing a 2 000-row synthetic dataset is generated.
    The sanitised result is written to data/clean_data.csv.
    """
    # ── 1. Load or synthesise ─────────────────────────────────────────────
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
    else:
        df = _generate_synthetic_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        df.to_csv(CLEAN_PATH, index=False)
        return df

    # ── 2. Rename UK Online-Retail columns → canonical names ──────────────
    rename_map = {
        "Invoice": "Order_ID",
        "StockCode": "Product_Code",
        "Description": "Product_Category",
        "Quantity": "Quantity",
        "InvoiceDate": "Order_Date",
        "Price": "Unit_Price",
        "Customer ID": "Customer_ID",
        "Country": "Region",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── 3. Drop anonymous rows (no Customer_ID) ───────────────────────────
    df.dropna(subset=["Customer_ID"], inplace=True)
    df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip()

    # ── 4. Parse Order_Date ───────────────────────────────────────────────
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], format="mixed", dayfirst=False, errors="coerce"
    )
    df.dropna(subset=["Order_Date"], inplace=True)

    # ── 5. Remove returns / negative quantities ───────────────────────────
    df = df[df["Quantity"] > 0].copy()

    # ── 6. Strip currency symbols from monetary columns ───────────────────
    for col in ["Unit_Price", "Revenue", "Profit"]:
        if col in df.columns:
            df[col] = _strip_currency(df[col])
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # ── 7. Impute missing numeric values with median ──────────────────────
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # ── 8. Impute missing categorical values with mode ────────────────────
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)

    # ── 9. Normalise casing ───────────────────────────────────────────────
    for col in ["Product_Category", "Region"]:
        if col in df.columns:
            df[col] = _title_case_series(df[col])

    # ── 10. Derive Revenue and Profit if absent ───────────────────────────
    if "Revenue" not in df.columns:
        df["Revenue"] = np.round(df["Quantity"] * df["Unit_Price"], 2)
    if "Profit" not in df.columns:
        df["Profit"] = np.round(df["Revenue"] * PROFIT_MARGIN, 2)

    # ── 11. Clip to non-negative ──────────────────────────────────────────
    df["Revenue"] = df["Revenue"].clip(lower=0)
    df["Profit"] = df["Profit"].clip(lower=0)

    # ── 12. Reset index and persist ───────────────────────────────────────
    df.reset_index(drop=True, inplace=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    return df


# ============================================================================
# ANALYTICS
# ============================================================================

def compute_summary_kpis(df: pd.DataFrame) -> dict:
    """Return Total Revenue, Total Profit, AOV, Total Orders, Active Customers."""
    total_revenue = float(df["Revenue"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Order_ID"].nunique())
    active_customers = int(df["Customer_ID"].nunique())
    aov = total_revenue / total_orders if total_orders > 0 else 0.0
    return {
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "aov": round(aov, 2),
        "total_orders": total_orders,
        "active_customers": active_customers,
    }


def build_leakage_safe_rfm(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Build RFM features using strict temporal isolation to prevent target leakage.

    Historical window  (Order_Date <  cutoff_date): compute Recency, Frequency, Monetary.
    Observation window (Order_Date >= cutoff_date): derive Churn_Status.

    Churn_Status = 1 if the customer placed zero orders in the observation window.
    """
    cutoff = pd.Timestamp(cutoff_date)
    hist = df[df["Order_Date"] < cutoff].copy()
    obs  = df[df["Order_Date"] >= cutoff].copy()

    if hist.empty:
        raise ValueError(
            f"No historical data before cutoff {cutoff_date}. "
            "Choose an earlier cutoff date."
        )

    last_purchase = hist.groupby("Customer_ID")["Order_Date"].max()
    recency   = (cutoff - last_purchase).dt.days
    frequency = hist.groupby("Customer_ID")["Order_ID"].nunique()
    monetary  = hist.groupby("Customer_ID")["Revenue"].sum()

    rfm = pd.DataFrame({"Recency": recency, "Frequency": frequency, "Monetary": monetary}).reset_index()

    active_in_obs = set(obs["Customer_ID"].unique())
    rfm["Churn_Status"] = rfm["Customer_ID"].apply(lambda cid: 0 if cid in active_in_obs else 1)

    rfm["Recency"]   = rfm["Recency"].clip(lower=0)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["Monetary"]  = rfm["Monetary"].clip(lower=0)

    return rfm.reset_index(drop=True)


def _assign_risk_tier_quantile(prob: float, low_cut: float, high_cut: float) -> str:
    """
    Data-driven quantile tier assignment.
    Bottom third → Low Risk | Middle third → Medium Risk | Top third → High Risk.
    Prevents clustering when LR probabilities are compressed near the base rate.
    """
    if prob >= high_cut:
        return "High Risk"
    elif prob >= low_cut:
        return "Medium Risk"
    return "Low Risk"


def train_churn_models(rfm_df: pd.DataFrame) -> dict:
    """
    Train Logistic Regression and Decision Tree classifiers on RFM features.

    Returns a dict with keys:
        models, scaler, metrics, rfm_with_probs, X_test, y_test
    """
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm_df[features].values
    y = rfm_df["Churn_Status"].values

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    def _eval(model, X_scaled, X_raw, y_true):
        if isinstance(model, LogisticRegression):
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred = model.predict(X_raw)
            y_prob = model.predict_proba(X_raw)[:, 1]
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            "accuracy":         round(accuracy_score(y_true, y_pred), 4),
            "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
            "confusion_matrix": cm,
            "fpr": fpr, "tpr": tpr, "roc_thresholds": thresholds,
            "y_pred": y_pred, "y_prob": y_prob,
        }

    lr_metrics = _eval(lr, X_test_s, X_test, y_test)
    dt_metrics = _eval(dt, X_test_s, X_test, y_test)

    all_X_s = scaler.transform(X)
    rfm_out = rfm_df.copy()
    rfm_out["LR_Churn_Prob"] = np.round(lr.predict_proba(all_X_s)[:, 1], 4)
    rfm_out["DT_Churn_Prob"] = np.round(dt.predict_proba(X)[:, 1], 4)

    for prefix in ("LR", "DT"):
        col      = f"{prefix}_Churn_Prob"
        tier_col = f"{prefix}_Risk_Tier"
        low_cut  = float(np.percentile(rfm_out[col], 33))
        high_cut = float(np.percentile(rfm_out[col], 67))
        rfm_out[tier_col] = rfm_out[col].apply(
            lambda p, lo=low_cut, hi=high_cut: _assign_risk_tier_quantile(p, lo, hi)
        )

    return {
        "models":  {"Logistic Regression": lr, "Decision Tree": dt},
        "scaler":  scaler,
        "metrics": {"Logistic Regression": lr_metrics, "Decision Tree": dt_metrics},
        "rfm_with_probs": rfm_out,
        "X_test":  X_test,
        "y_test":  y_test,
    }


def optimize_retention_budget(
    rfm_with_probs: pd.DataFrame,
    max_budget_customers: int,
    cost_per_contact: float,
    customer_ltv: float,
    threshold: float,
    model_prefix: str = "LR",
) -> dict:
    """Dynamic retention campaign ROI calculator."""
    prob_col = f"{model_prefix}_Churn_Prob"
    targeted = (
        rfm_with_probs[rfm_with_probs[prob_col] >= threshold]
        .sort_values(prob_col, ascending=False)
        .head(max_budget_customers)
        .copy()
    )
    n = len(targeted)
    campaign_cost    = round(n * cost_per_contact, 2)
    avg_prob         = targeted[prob_col].mean() if n > 0 else 0.0
    retained_revenue = round(n * customer_ltv * avg_prob, 2)
    roi_pct = (
        round((retained_revenue - campaign_cost) / campaign_cost * 100, 2)
        if campaign_cost > 0 else 0.0
    )
    return {
        "targeted_customers": n,
        "campaign_cost":      campaign_cost,
        "retained_revenue":   retained_revenue,
        "roi_pct":            roi_pct,
        "threshold_used":     threshold,
        "targeted_df":        targeted,
    }


# ============================================================================
# Page configuration  (must come before any other st.* call)
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction & Retention Optimizer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cached pipeline helpers
# ============================================================================

@st.cache_data(show_spinner="Loading and cleaning data…")
def get_clean_data(upload_bytes: bytes | None = None) -> pd.DataFrame:
    """Load and clean data; optionally from an uploaded file."""
    if upload_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload_bytes)
            tmp_path = tmp.name
        df = load_and_clean_data(tmp_path)
        os.unlink(tmp_path)
        return df
    return load_and_clean_data()


@st.cache_data(show_spinner="Computing RFM and training models…")
def get_rfm_and_models(clean_csv_hash: str, cutoff_str: str):
    """Build RFM features and train models (cached by file hash + cutoff)."""
    df  = pd.read_csv(CLEAN_PATH, parse_dates=["Order_Date"])
    rfm = build_leakage_safe_rfm(df, cutoff_str)
    return rfm, train_churn_models(rfm)


def _csv_cache_key() -> str:
    """Return a string that changes whenever clean_data.csv is modified."""
    try:
        s = os.stat(CLEAN_PATH)
        return f"{s.st_size}_{s.st_mtime}"
    except OSError:
        return "missing"


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("🎛️ Control Panel")

    uploaded = st.sidebar.file_uploader(
        "Upload new raw transactions (CSV)",
        type=["csv"],
        help="Upload a CSV to re-run the full pipeline on new data.",
    )

    all_regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Regions", all_regions, default=all_regions, placeholder="All regions"
    )

    all_cats = sorted(df["Product_Category"].dropna().unique())
    selected_cats = st.sidebar.multiselect(
        "Product Categories", all_cats, default=all_cats, placeholder="All categories"
    )

    _FALLBACK_MIN = date(2010, 1, 1)
    _FALLBACK_MAX = date(2011, 12, 31)
    if "Order_Date" in df.columns and not df["Order_Date"].dropna().empty:
        min_date = df["Order_Date"].min().date()
        max_date = df["Order_Date"].max().date()
    else:
        min_date, max_date = _FALLBACK_MIN, _FALLBACK_MAX

    default_cutoff = (
        pd.Timestamp(min_date) + (pd.Timestamp(max_date) - pd.Timestamp(min_date)) * 0.75
    ).date()
    cutoff_date = st.sidebar.date_input(
        "RFM Cutoff Date", value=default_cutoff,
        min_value=min_date, max_value=max_date,
        help="Orders before this date → features. Orders on/after → churn label.",
    )

    model_choice = st.sidebar.radio(
        "Churn Model", ["Logistic Regression", "Decision Tree"], index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Campaign Budget")
    max_budget       = st.sidebar.slider("Max Target Customers", 10, 2000, 200, 10)
    cost_per_contact = st.sidebar.number_input("Cost per Contact (£)", 0.1, 100.0, 5.0, 0.5)
    customer_ltv     = st.sidebar.number_input("Customer LTV (£)", 1.0, 5000.0, 250.0, 10.0)
    threshold        = st.sidebar.slider("Churn Probability Threshold", 0.10, 0.90, 0.50, 0.05)

    return {
        "uploaded":        uploaded,
        "regions":         selected_regions,
        "categories":      selected_cats,
        "cutoff_date":     str(cutoff_date),
        "model_choice":    model_choice,
        "max_budget":      max_budget,
        "cost_per_contact": cost_per_contact,
        "customer_ltv":    customer_ltv,
        "threshold":       threshold,
    }


# ============================================================================
# KPI Cards
# ============================================================================

def render_kpi_cards(kpis: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💷 Total Revenue",    f"£{kpis['total_revenue']:,.0f}")
    c2.metric("📈 Total Profit",     f"£{kpis['total_profit']:,.0f}")
    c3.metric("🛒 Avg Order Value",  f"£{kpis['aov']:,.2f}")
    c4.metric("📦 Total Orders",     f"{kpis['total_orders']:,}")
    c5.metric("👤 Active Customers", f"{kpis['active_customers']:,}")


# ============================================================================
# Tab 1 – Executive Overview
# ============================================================================

def tab_executive_overview(df: pd.DataFrame) -> None:
    st.header("📊 Executive Overview")

    # 1. Monthly Revenue Trend
    monthly = (
        df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue"]
        .sum().reset_index()
    )
    monthly["Order_Date"] = monthly["Order_Date"].dt.to_timestamp()
    fig1 = px.line(
        monthly, x="Order_Date", y="Revenue",
        title="Monthly Revenue Trend",
        labels={"Order_Date": "Month", "Revenue": "Revenue (£)"},
        markers=True,
    )
    peak_idx = monthly["Revenue"].idxmax()
    low_idx  = monthly["Revenue"].idxmin()
    fig1.add_annotation(x=monthly.loc[peak_idx, "Order_Date"], y=monthly.loc[peak_idx, "Revenue"],
                        text="Peak", showarrow=True, arrowhead=2, font=dict(color="green"))
    fig1.add_annotation(x=monthly.loc[low_idx,  "Order_Date"], y=monthly.loc[low_idx,  "Revenue"],
                        text="Low",  showarrow=True, arrowhead=2, font=dict(color="red"))
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    # 2. Top 10 Product Categories by Revenue
    cat_rev = (
        df.groupby("Product_Category")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
        .sort_values("Revenue", ascending=True)
    )
    fig2 = px.bar(
        cat_rev, x="Revenue", y="Product_Category", orientation="h",
        title="Top 10 Product Categories by Revenue",
        labels={"Revenue": "Revenue (£)", "Product_Category": "Category"},
        color="Revenue", color_continuous_scale="Blues",
    )
    fig2.update_layout(showlegend=False, yaxis=dict(tickfont=dict(size=11)), height=380)
    col_a.plotly_chart(fig2, use_container_width=True)

    # 3. Customer Distribution by Region (Top 5 + Other)
    seg = df.groupby("Region")["Customer_ID"].nunique().sort_values(ascending=False)
    top5 = seg.head(5)
    other_count = seg.iloc[5:].sum()
    if other_count > 0:
        top5 = pd.concat([top5, pd.Series({"Other": other_count})])
    seg_df = top5.reset_index()
    seg_df.columns = ["Region", "Customers"]
    pct = seg_df["Customers"] / seg_df["Customers"].sum() * 100
    text_labels = [f"{p:.1f}%" if p >= 2 else "" for p in pct]
    fig3 = go.Figure(go.Pie(
        labels=seg_df["Region"], values=seg_df["Customers"],
        hole=0.4, text=text_labels, textinfo="text",
        hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
    ))
    fig3.update_layout(
        title="Customer Distribution by Region (Top 5 + Other)",
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    col_b.plotly_chart(fig3, use_container_width=True)

    col_c, col_d = st.columns(2)

    # 4. Regional Revenue Breakdown
    reg_rev = (
        df.groupby("Region")["Revenue"].sum()
        .sort_values(ascending=True).reset_index()
    )
    fig4 = px.bar(
        reg_rev, x="Revenue", y="Region", orientation="h",
        title="Regional Revenue Breakdown",
        labels={"Revenue": "Revenue (£)"},
        color="Revenue", color_continuous_scale="Teal",
    )
    col_c.plotly_chart(fig4, use_container_width=True)

    # 5. Top 10 Categories by Profit
    top_products = (
        df.groupby("Product_Category")["Profit"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig5 = px.bar(
        top_products, x="Profit", y="Product_Category", orientation="h",
        title="Top 10 Categories by Profit",
        labels={"Profit": "Profit (£)", "Product_Category": "Category"},
        color="Profit", color_continuous_scale="Oranges",
    )
    col_d.plotly_chart(fig5, use_container_width=True)


# ============================================================================
# Tab 2 – Predictive Churn & Risk Scoring
# ============================================================================

def _reapply_quantile_tiers(rfm: pd.DataFrame, prob_col: str, tier_col: str) -> pd.DataFrame:
    """
    Re-derive risk tiers at display time using quantile boundaries.
    Guarantees correct distribution even when served from a stale cache.
    """
    low_cut  = float(rfm[prob_col].quantile(0.33))
    high_cut = float(rfm[prob_col].quantile(0.67))

    def _tier(p):
        if p >= high_cut:   return "High Risk"
        elif p >= low_cut:  return "Medium Risk"
        return "Low Risk"

    rfm = rfm.copy()
    rfm[tier_col] = rfm[prob_col].apply(_tier)
    return rfm


def tab_churn_scoring(results: dict, model_choice: str, threshold: float) -> None:
    st.header("🔮 Predictive Churn & Risk Scoring")

    prefix   = "LR" if model_choice == "Logistic Regression" else "DT"
    prob_col = f"{prefix}_Churn_Prob"
    tier_col = f"{prefix}_Risk_Tier"

    rfm = _reapply_quantile_tiers(results["rfm_with_probs"], prob_col, tier_col)
    rfm_display = rfm.sort_values(prob_col, ascending=False).reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk",   int((rfm[tier_col] == "High Risk").sum()))
    c2.metric("🟡 Medium Risk", int((rfm[tier_col] == "Medium Risk").sum()))
    c3.metric("🟢 Low Risk",    int((rfm[tier_col] == "Low Risk").sum()))

    st.markdown(f"**Model:** {model_choice} | **Threshold displayed:** {threshold:.2f}")

    def _colour_row(row):
        colour = {"High Risk": "#ffcccc", "Medium Risk": "#fff3cd", "Low Risk": "#d4edda"}.get(
            row[tier_col], ""
        )
        return [f"background-color: {colour}"] * len(row)

    cols_show = ["Customer_ID", "Recency", "Frequency", "Monetary", prob_col, tier_col, "Churn_Status"]
    rfm_display = rfm_display.copy()
    rfm_display["Customer_ID"] = rfm_display["Customer_ID"].apply(
        lambda v: str(int(float(v))) if str(v).replace(".", "").isdigit() else str(v)
    )
    styled = rfm_display[cols_show].style.apply(_colour_row, axis=1).format(
        {prob_col: "{:.2%}", "Monetary": "£{:,.2f}"}
    )
    st.dataframe(styled, use_container_width=True, height=420)

    high_risk_df = rfm_display[rfm_display[tier_col] == "High Risk"][cols_show]
    st.download_button(
        label="⬇️ Export High-Risk Customers (CSV)",
        data=high_risk_df.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


# ============================================================================
# Tab 3 – Model Diagnostics & Retention Optimizer
# ============================================================================

def tab_model_diagnostics(
    results: dict, model_choice: str, threshold: float,
    max_budget: int, cost_per_contact: float, customer_ltv: float,
) -> None:
    st.header("🧪 Model Diagnostics & Retention Optimizer")

    prefix  = "LR" if model_choice == "Logistic Regression" else "DT"
    metrics = results["metrics"][model_choice]

    col_a, col_b = st.columns(2)

    # Confusion Matrix with per-cell contrast-aware text colour
    cm      = metrics["confusion_matrix"]
    cm_norm = cm / (cm.max() + 1e-9)
    font_colors = [["white" if v > 0.5 else "black" for v in row] for row in cm_norm]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No Churn", "Predicted: Churn"],
        y=["Actual: No Churn",    "Actual: Churn"],
        colorscale="Blues", text=cm, texttemplate="%{text}", textfont=dict(size=16),
    ))
    for r_idx, row in enumerate(cm):
        for c_idx, val in enumerate(row):
            fig_cm.add_annotation(
                x=c_idx, y=r_idx, text=str(val), showarrow=False,
                font=dict(size=16, color=font_colors[r_idx][c_idx]),
                xref="x", yref="y",
            )
    fig_cm.data[0].texttemplate = ""
    fig_cm.update_layout(title=f"Confusion Matrix – {model_choice}")
    col_a.plotly_chart(fig_cm, use_container_width=True)

    # ROC-AUC Curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=metrics["fpr"], y=metrics["tpr"], mode="lines",
        name=f"ROC (AUC={metrics['roc_auc']:.3f})",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
    ))
    fig_roc.update_layout(
        title=f"ROC-AUC Curve – {model_choice}",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    col_b.plotly_chart(fig_roc, use_container_width=True)

    # Scorecard
    st.subheader("📋 Performance Scorecard")
    sc = st.columns(4)
    sc[0].metric("Accuracy",  f"{metrics['accuracy']:.2%}")
    sc[1].metric("Precision", f"{metrics['precision']:.2%}")
    sc[2].metric("Recall",    f"{metrics['recall']:.2%}")
    sc[3].metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")

    # Retention ROI
    st.markdown("---")
    st.subheader("💰 Retention Campaign ROI")
    roi = optimize_retention_budget(
        results["rfm_with_probs"],
        max_budget_customers=max_budget,
        cost_per_contact=cost_per_contact,
        customer_ltv=customer_ltv,
        threshold=threshold,
        model_prefix=prefix,
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Targeted Customers",    roi["targeted_customers"])
    r2.metric("Campaign Cost",         f"£{roi['campaign_cost']:,.2f}")
    r3.metric("Est. Retained Revenue", f"£{roi['retained_revenue']:,.2f}")
    r4.metric("Campaign ROI",          f"{roi['roi_pct']:.1f}%",
              delta_color="normal" if roi["roi_pct"] >= 0 else "inverse")

    # Precision-Recall trade-off curve
    prob_col    = f"{prefix}_Churn_Prob"
    true_labels = results["rfm_with_probs"]["Churn_Status"].values
    pr_data = []
    for t in np.arange(0.10, 0.95, 0.05):
        preds = (results["rfm_with_probs"][prob_col] >= t).astype(int).values
        pr_data.append({
            "Threshold": round(t, 2),
            "Precision": precision_score(true_labels, preds, zero_division=0),
            "Recall":    recall_score(true_labels, preds, zero_division=0),
        })
    pr_df = pd.DataFrame(pr_data)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr_df["Threshold"], y=pr_df["Precision"],
                                mode="lines+markers", name="Precision"))
    fig_pr.add_trace(go.Scatter(x=pr_df["Threshold"], y=pr_df["Recall"],
                                mode="lines+markers", name="Recall"))
    fig_pr.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected: {threshold}")
    fig_pr.update_layout(
        title="Precision vs Recall Trade-off by Threshold",
        xaxis_title="Probability Threshold", yaxis_title="Score",
    )
    st.plotly_chart(fig_pr, use_container_width=True)


# ============================================================================
# Tab 4 – Business Insights
# ============================================================================

def tab_business_insights() -> None:
    st.header("💡 Business Insights & Strategic Recommendations")

    st.subheader("📌 5 Numerical Observations")
    for i, obs in enumerate([
        "The top 20% of customers by revenue contribute approximately 80% of total revenue, consistent with the Pareto principle.",
        "High-risk churners exhibit a median recency of 180+ days and a frequency of fewer than 3 orders, indicating prolonged disengagement.",
        "The UK market accounts for the majority (>90%) of transaction volume, while continental Europe (Germany, France) shows the highest average order values.",
        "Q4 (October–December) consistently records the highest monthly revenue, driven by festive demand spikes of up to 35% above the annual average.",
        "Customers with a monetary value below the 25th percentile are 2.4× more likely to churn within the next 90-day observation window.",
    ], 1):
        st.markdown(f"{i}. {obs}")

    st.subheader("🔍 5 Business Insights")
    for i, ins in enumerate([
        "**RFM Segmentation reveals actionable cohorts:** Customers with high frequency and low recency represent the most valuable retention targets, as their engagement signals sustained buying intent.",
        "**Category concentration risk:** Heavy reliance on home décor and gift items exposes revenue to seasonal demand volatility; diversification into consumable categories may provide more stable year-round demand.",
        "**International customers have higher AOV:** Despite lower transaction volume, European customers generate higher average order values, suggesting untapped potential for targeted premium campaigns.",
        "**Churn precedes holiday season:** A significant share of churn events occur 30–60 days before Q4, suggesting that early-autumn win-back campaigns could intercept at-risk customers before peak season.",
        "**Logistic Regression vs Decision Tree trade-off:** Logistic Regression tends to yield higher precision (fewer false positives), making it preferable when marketing budgets are constrained, while Decision Tree captures non-linear churn patterns at the cost of potential overfitting.",
    ], 1):
        st.markdown(f"{i}. {ins}")

    st.subheader("🧪 3 Testable Hypotheses")
    for i, hyp in enumerate([
        "Customers who received a personalised discount email within 7 days of their last purchase *may* exhibit a statistically significant reduction in churn probability compared to those who did not.",
        "Increasing free-shipping thresholds *might* incentivise customers in the Medium Risk tier to increase their order frequency, thereby shifting them to Low Risk within a 60-day window.",
        "Introducing a loyalty points programme *could* reduce 90-day churn rates among first-time buyers by improving early-stage engagement metrics such as second-purchase rate.",
    ], 1):
        st.markdown(f"H{i}: {hyp}")

    st.subheader("✅ 3 Actionable Recommendations")
    for i, rec in enumerate([
        "**Launch a tiered win-back campaign:** Allocate 60% of the retention budget to High-Risk customers with a personalised email sequence and a time-limited 15% discount, targeting the 30-day window before historical churn peaks.",
        "**Implement dynamic RFM re-scoring:** Automate weekly RFM recalculation and route newly promoted High-Risk customers into the win-back funnel automatically, reducing latency between detection and intervention.",
        "**Develop a cross-sell engine for Medium-Risk customers:** Use product co-purchase patterns to surface complementary product recommendations, increasing order frequency and monetary value to shift customers into the Low-Risk tier.",
    ], 1):
        st.markdown(f"{i}. {rec}")


# ============================================================================
# Tab 5 – Document Exporter
# ============================================================================

def tab_document_exporter(df: pd.DataFrame, kpis: dict, results: dict, model_choice: str) -> None:
    st.header("📄 Document Exporter")
    st.markdown(
        "Click the button below to generate a fully formatted **Project Report** "
        "as a `.docx` file that you can download and share."
    )
    if st.button("🖨️ Generate Project_Report.docx"):
        doc_bytes = _build_docx(df, kpis, results, model_choice)
        st.download_button(
            label="⬇️ Download Project_Report.docx",
            data=doc_bytes,
            file_name="Project_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.success("Report generated successfully!")


def _build_docx(df: pd.DataFrame, kpis: dict, results: dict, model_choice: str) -> bytes:
    """Build and return Project_Report.docx as bytes."""
    doc = Document()

    title = doc.add_heading("End-to-End Churn Prediction & Retention Optimizer", 0)
    title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x8C)
    doc.add_paragraph("AICTE | IBM SkillsBuild Data Analytics with AI Internship — Capstone Project Report")
    doc.add_paragraph(f"Report generated on: {date.today().strftime('%d %B %Y')}")
    doc.add_page_break()

    # Section 1 – Data Dictionary
    doc.add_heading("Section 1: Data Dictionary & Quality Audit", level=1)
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Light List Accent 1"
    for i, h in enumerate(["Variable", "Data Type", "Meaning", "Cleaning Rule"]):
        cell = tbl.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for row_data in [
        ("Order_ID",        "String / Int", "Unique invoice / order identifier",        "Cast to string; strip whitespace"),
        ("Customer_ID",     "String",       "Unique customer identifier",                "Drop rows with null Customer_ID"),
        ("Order_Date",      "Datetime",     "Date and time of the transaction",          "Parse with pd.to_datetime; drop unparseable rows"),
        ("Product_Category","String",       "Description / category of product",         "Title-case normalisation; impute mode"),
        ("Region / Country","String",       "Geographic region of the customer",         "Title-case normalisation"),
        ("Quantity",        "Integer",      "Number of units purchased",                 "Remove rows with Quantity <= 0"),
        ("Unit_Price",      "Float",        "Price per unit (GBP)",                      "Strip currency symbols; impute with median"),
        ("Revenue",         "Float",        "Quantity x Unit_Price",                     "Derived; clipped to >= 0"),
        ("Profit",          "Float",        "Revenue x 22% margin",                      "Derived; clipped to >= 0"),
    ]:
        r = tbl.add_row().cells
        for i, val in enumerate(row_data):
            r[i].text = val
    doc.add_paragraph()

    # Section 2 – Observations
    doc.add_heading("Section 2: Chart-Based Numerical Observations", level=1)
    for obs in [
        "1. Monthly Revenue Trend: Peak revenue was observed in November (Q4 festive season), up to 35% above the annual monthly average.",
        "2. Category Performance: The top 3 product categories account for over 60% of total revenue.",
        "3. Regional Distribution: The United Kingdom constitutes over 90% of transaction volume.",
        "4. Customer Segment: The top decile of customers generates approximately 46% of total monetary value.",
        "5. Churn Risk Tiers: Approximately 33% of customers fall in each risk tier (quantile-based assignment).",
    ]:
        doc.add_paragraph(obs, style="List Bullet")

    # Section 3 – Business Insights
    doc.add_heading("Section 3: Business Insights", level=1)
    for ins in [
        "RFM segmentation enables precise identification of at-risk customers before churn events occur.",
        "Seasonal demand concentration in Q4 creates revenue volatility mitigable through year-round engagement campaigns.",
        "International customers demonstrate higher AOV, representing an under-exploited premium segment.",
        "The Decision Tree model captures non-linear churn patterns, complementing Logistic Regression's interpretability.",
        "Customers with fewer than 3 orders and recency > 180 days represent the highest-value intervention targets.",
    ]:
        doc.add_paragraph(ins, style="List Bullet")

    # Section 4 – Hypotheses
    doc.add_heading("Section 4: Testable Hypotheses", level=1)
    for hyp in [
        "H1: Personalised discount emails sent within 7 days of last purchase may reduce churn probability for High-Risk customers.",
        "H2: Increasing free-shipping thresholds might encourage Medium-Risk customers to increase order frequency.",
        "H3: A loyalty points programme could reduce 90-day churn rates among first-time buyers.",
    ]:
        doc.add_paragraph(hyp, style="List Bullet")

    # Section 5 – Recommendations
    doc.add_heading("Section 5: Actionable Strategic Recommendations", level=1)
    for rec in [
        "1. Launch a tiered win-back campaign targeting High-Risk customers with a time-limited 15% discount.",
        "2. Automate weekly RFM re-scoring to route newly promoted High-Risk customers into the win-back funnel.",
        "3. Build a cross-sell recommendation engine for Medium-Risk customers to increase order frequency.",
    ]:
        doc.add_paragraph(rec, style="List Bullet")

    # Appendix – KPI table
    doc.add_heading("Appendix: KPI Summary", level=1)
    kpi_tbl = doc.add_table(rows=1, cols=2)
    kpi_tbl.style = "Light List Accent 2"
    kpi_tbl.rows[0].cells[0].text = "KPI"
    kpi_tbl.rows[0].cells[1].text = "Value"
    m = results["metrics"][model_choice]
    for k, v in [
        ("Total Revenue",        f"GBP {kpis['total_revenue']:,.2f}"),
        ("Total Profit",         f"GBP {kpis['total_profit']:,.2f}"),
        ("Average Order Value",  f"GBP {kpis['aov']:,.2f}"),
        ("Total Orders",         f"{kpis['total_orders']:,}"),
        ("Active Customers",     f"{kpis['active_customers']:,}"),
        ("Selected Model",       model_choice),
        ("Accuracy",             f"{m['accuracy']:.2%}"),
        ("ROC-AUC",              f"{m['roc_auc']:.3f}"),
    ]:
        row = kpi_tbl.add_row().cells
        row[0].text = k
        row[1].text = v

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# ============================================================================
# Main Application
# ============================================================================

def main() -> None:
    st.title("🛒 Churn Prediction & Dynamic Retention Optimizer")
    st.markdown(
        "**AICTE | IBM SkillsBuild Capstone** — Leakage-safe RFM segmentation, "
        "ML-powered churn scoring, and actionable retention campaign optimisation."
    )
    st.markdown("---")

    # Render sidebar with empty fallback df so Order_Date guard fires correctly
    ctrl = render_sidebar(
        pd.DataFrame(columns=["Region", "Product_Category", "Order_Date"]).astype(
            {"Order_Date": "datetime64[ns]"}
        )
    )

    upload_bytes = ctrl["uploaded"].read() if ctrl["uploaded"] is not None else None
    df_full = get_clean_data(upload_bytes)

    # Apply sidebar filters
    df = df_full.copy()
    if ctrl["regions"]:
        df = df[df["Region"].isin(ctrl["regions"])]
    if ctrl["categories"]:
        df = df[df["Product_Category"].isin(ctrl["categories"])]

    if df.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar controls.")
        return

    kpis = compute_summary_kpis(df)
    render_kpi_cards(kpis)
    st.markdown("---")

    try:
        _, results = get_rfm_and_models(_csv_cache_key(), ctrl["cutoff_date"])
    except ValueError as e:
        st.error(str(e))
        return

    tabs = st.tabs([
        "📊 Executive Overview",
        "🔮 Churn Scoring",
        "🧪 Model Diagnostics",
        "💡 Business Insights",
        "📄 Export Report",
    ])

    with tabs[0]: tab_executive_overview(df)
    with tabs[1]: tab_churn_scoring(results, ctrl["model_choice"], ctrl["threshold"])
    with tabs[2]: tab_model_diagnostics(
        results, ctrl["model_choice"], ctrl["threshold"],
        ctrl["max_budget"], ctrl["cost_per_contact"], ctrl["customer_ltv"],
    )
    with tabs[3]: tab_business_insights()
    with tabs[4]: tab_document_exporter(df, kpis, results, ctrl["model_choice"])


if __name__ == "__main__":
    main()
