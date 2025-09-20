
import io
import os
import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PAL Radiators – Inventory Intelligence", layout="wide")

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def load_excel_from_path(path: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    data = {}
    for s in xls.sheet_names:
        data[s] = pd.read_excel(path, sheet_name=s)
    return data

@st.cache_data(show_spinner=False)
def load_excel_from_buffer(buf) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(buf)
    data = {}
    for s in xls.sheet_names:
        data[s] = pd.read_excel(buf, sheet_name=s)
    return data

def coalesce_cols(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def simple_exp_smooth(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if len(x) == 0:
        return x
    s = np.zeros_like(x, dtype=float)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
    return s

def abc_classification(series: pd.Series, A=0.8, B=0.95):
    total = series.sum()
    if total <= 0:
        return pd.Series(index=series.index, dtype="object")
    cum = series.sort_values(ascending=False).cumsum() / total
    cls = pd.Series(index=series.index, dtype="object")
    cls[cum <= A] = "A"
    cls[(cum > A) & (cum <= B)] = "B"
    cls[cum > B] = "C"
    return cls

def weighted_avg(series_values, weights):
    v = np.array(series_values, dtype=float)
    w = np.array(weights, dtype=float)
    mask = ~np.isnan(v) & ~np.isnan(w)
    if mask.sum() == 0 or w[mask].sum() == 0:
        return np.nan
    return (v[mask] * w[mask]).sum() / w[mask].sum()

# ------------------------------
# Sidebar: Choose data source
# ------------------------------
st.sidebar.header("📦 Data Source")
st.sidebar.write("Upload Excel OR read from a file committed in your GitHub repo (e.g. `data/pal_radiators.xlsx`).")
uploaded = st.sidebar.file_uploader("Upload Excel (must include 'Purchases' and 'Sales' sheets)", type=["xlsx"])
repo_path = st.sidebar.text_input("…or read from repo path", value="data/PAL Radiator Purchase and Sold - 2022 onwards.xlsx")
open_btn = st.sidebar.button("Load Data")
# File presence check
if repo_path:
    if os.path.exists(repo_path):
        try:
            size_mb = os.path.getsize(repo_path) / (1024*1024)
            st.sidebar.success(f"✅ Found: {repo_path} ({size_mb:.1f} MB)")
        except Exception:
            st.sidebar.success(f"✅ Found: {repo_path}")
    else:
        st.sidebar.error(f"❌ Not found on server: {repo_path}")

# Tiny check for repo path existence
if repo_path:
    if os.path.exists(repo_path):
        st.sidebar.success(f"✔ Repo file found: {repo_path}")
    else:
        st.sidebar.error(f"✘ Repo file not found: {repo_path}")


excel_data = None
errs = []

if uploaded is not None and open_btn:
    try:
        excel_data = load_excel_from_buffer(uploaded)
        st.sidebar.success("Loaded from uploaded file.")
    except Exception as e:
        errs.append(f"Upload failed: {e}")

if excel_data is None and open_btn:
    if repo_path and os.path.exists(repo_path):
        try:
            excel_data = load_excel_from_path(repo_path)
            st.sidebar.success(f"Loaded from repo path: {repo_path}")
        except Exception as e:
            errs.append(f"Repo path failed: {e}")
    else:
        if repo_path:
            errs.append(f"Repo path not found on server: {repo_path}")

if uploaded is None and not (repo_path and os.path.exists(repo_path)):
    st.info("Upload your Excel on the left **or** commit it to your GitHub repo and provide its relative path, then click **Load Data**.")

for m in errs:
    st.sidebar.error(m)

if not excel_data:
    st.stop()

if "Purchases" not in excel_data or "Sales" not in excel_data:
    st.error("Excel must contain sheets named 'Purchases' and 'Sales'.")
    st.stop()

purch = excel_data["Purchases"].copy()
sales = excel_data["Sales"].copy()

# Normalize columns
purch.columns = [c.strip() for c in purch.columns]
sales.columns = [c.strip() for c in sales.columns]

# Key columns
def pick(df, *names):
    col = coalesce_cols(df, names)
    return col

col_item_p = pick(purch, "Item Number", "Item", "SKU", "Product Code")
col_item_s = pick(sales, "Item Number", "Item", "SKU", "Product Code")
col_item = col_item_p

col_date_p = pick(purch, "Date", "Posting Date", "Invoice Date")
col_date_s = pick(sales, "Date", "Posting Date", "Invoice Date")

col_qty_p = pick(purch, "Quantity", "Qty", "QTY")
col_qty_s = pick(sales, "Quantity", "Qty", "QTY")

col_cost = "Unit Price" if "Unit Price" in purch.columns else pick(purch, "Unit Cost", "Cost", "UnitPrice")
col_amt = "Amount" if "Amount" in sales.columns else pick(sales, "Net Amount", "Line Amount")

col_vendor = "Vendor Name" if "Vendor Name" in purch.columns else pick(purch, "Vendor", "Supplier")
col_customer = "Customer Name" if "Customer Name" in sales.columns else pick(sales, "Customer", "Cust Name")

col_product_name = "Product Name" if "Product Name" in sales.columns else col_item_s
# Schema diagnostics
required_sales = {
    "Item": (col_item_s, ("Item Number","Item","SKU","Product Code")),
    "Date": (col_date_s, ("Date","Posting Date","Invoice Date")),
    "Quantity": (col_qty_s, ("Quantity","Qty","QTY")),
    "Amount": (col_amt, ("Amount","Net Amount","Line Amount")),
}
required_purch = {
    "Item": (col_item_p, ("Item Number","Item","SKU","Product Code")),
    "Date": (col_date_p, ("Date","Posting Date","Invoice Date")),
    "Quantity": (col_qty_p, ("Quantity","Qty","QTY")),
    "Unit Cost": (col_cost, ("Unit Price","Unit Cost","Cost","UnitPrice")),
}

missing_msgs = []
for label, (found, candidates) in required_sales.items():
    if found is None:
        missing_msgs.append(f"Sales sheet missing **{label}**. Expected one of: {', '.join(candidates)}")

for label, (found, candidates) in required_purch.items():
    if found is None:
        missing_msgs.append(f"Purchases sheet missing **{label}**. Expected one of: {', '.join(candidates)}")

with st.expander("🔧 Schema diagnostics (click to expand)"):
    st.write("**Sales columns found:**", list(sales.columns))
    st.write("**Purchases columns found:**", list(purch.columns))
    st.write("**Key column mapping (None = not found):**")
    st.json({
        "Sales": {"Item": col_item_s, "Date": col_date_s, "Quantity": col_qty_s, "Amount": col_amt, "Customer": col_customer, "ProductName": col_product_name},
        "Purchases": {"Item": col_item_p, "Date": col_date_p, "Quantity": col_qty_p, "UnitCost": col_cost, "Vendor": col_vendor}
    })

if missing_msgs:
    st.error("I couldn't find some required columns. Please rename columns in your Excel or use the 'Upload' option.

" + "

".join(missing_msgs))
    st.stop()

col_invoice_s = pick(sales, "Invoice", "Invoice No", "Invoice Number", "Sales Invoice")

# Types
for df, cdate in [(purch, col_date_p), (sales, col_date_s)]:
    df[cdate] = pd.to_datetime(df[cdate], errors="coerce")

for df, cqty in [(purch, col_qty_p), (sales, col_qty_s)]:
    df[cqty] = pd.to_numeric(df[cqty], errors="coerce").fillna(0.0)

purch[col_cost] = pd.to_numeric(purch.get(col_cost, np.nan), errors="coerce").fillna(0.0)
sales[col_amt] = pd.to_numeric(sales.get(col_amt, np.nan), errors="coerce").fillna(0.0)

# ------------------------------
# Filters
# ------------------------------
st.sidebar.header("🔎 Filters")
min_date = min(purch[col_date_p].min(), sales[col_date_s].min())
max_date = max(purch[col_date_p].max(), sales[col_date_s].max())
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date() if pd.notnull(min_date) else None,
           max_date.date() if pd.notnull(max_date) else None)
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date, end_date = min_date, max_date

purch = purch[(purch[col_date_p] >= start_date) & (purch[col_date_p] <= end_date)]
sales = sales[(sales[col_date_s] >= start_date) & (sales[col_date_s] <= end_date)]

# Opening stock upload
st.sidebar.header("📦 Opening Stock (optional)")
st.sidebar.write("Upload CSV with columns: **Item Number, OpeningQty** to refine stock.")
opening_file = st.sidebar.file_uploader("Opening Stock CSV", type=["csv"], key="opening")
default_opening = st.sidebar.number_input("Default opening per item (if missing)", min_value=0.0, value=0.0, step=1.0)

opening = None
if opening_file is not None:
    try:
        opening = pd.read_csv(opening_file)
        opening.columns = [c.strip() for c in opening.columns]
    except Exception as e:
        st.sidebar.error(f"Opening stock CSV error: {e}")

# Lead times & min/max upload
st.sidebar.header("🧭 Per-SKU Lead Time & Levels (optional)")
st.sidebar.write("Upload CSV with columns: **Item Number, LeadTimeDays, MinLevel, MaxLevel**")
lt_file = st.sidebar.file_uploader("LeadTime/Levels CSV", type=["csv"], key="ltmm")
ltmm = None
if lt_file is not None:
    try:
        ltmm = pd.read_csv(lt_file)
        ltmm.columns = [c.strip() for c in ltmm.columns]
    except Exception as e:
        st.sidebar.error(f"LeadTime/Levels CSV error: {e}")

# Optional payments/receipts upload for AR aging
st.sidebar.header("💸 Payments (optional, for true AR aging)")
st.sidebar.write("Upload CSV with columns: **Customer Name, Invoice, Amount, Date**")
pay_file = st.sidebar.file_uploader("Payments CSV", type=["csv"], key="payments")
payments = None
if pay_file is not None:
    try:
        payments = pd.read_csv(pay_file)
        payments.columns = [c.strip() for c in payments.columns]
        if "Date" in payments.columns:
            payments["Date"] = pd.to_datetime(payments["Date"], errors="coerce")
    except Exception as e:
        st.sidebar.error(f"Payments CSV error: {e}")

# Parameters
lead_time_days_default = st.sidebar.number_input("Default lead time (days)", min_value=1, value=90, step=1)
service_level_z = st.sidebar.number_input("Service level Z-score (e.g., 1.28 ≈ 90%)", min_value=0.0, value=1.28, step=0.01)

# ------------------------------
# Aggregations & core computations
# ------------------------------
# Sales unit price estimate
sales["UnitPrice_Est"] = sales[col_amt] / sales[col_qty_s].replace(0, np.nan)

# Monthly groups
# Robust purchase aggregation: qty, spend, weighted average cost (WAC)
_purch_with_month = purch.assign(_month=purch[col_date_p].dt.to_period("M").dt.to_timestamp())
purch_m = (
    _purch_with_month
    .groupby([col_item_p, "_month"], as_index=False)
    .apply(lambda g: pd.Series({
        "qty_p": g[col_qty_p].sum(),
        "spend": (g[col_qty_p] * g[col_cost]).sum(),
        "avg_cost": (g[col_qty_p] * g[col_cost]).sum() / g[col_qty_p].sum() if g[col_qty_p].sum() > 0 else np.nan
    }))
    .reset_index(drop=True)
    .rename(columns={col_item_p: col_item})
)

# Sales aggregation with weighted average unit price
_sales_with_month = sales.assign(_month=sales[col_date_s].dt.to_period("M").dt.to_timestamp())
sales_m = (
    _sales_with_month
    .groupby([col_item_s, "_month"], as_index=False)
    .apply(lambda g: pd.Series({
        "qty_s": g[col_qty_s].sum(),
        "revenue": g[col_amt].sum(),
        "avg_price": (g["UnitPrice_Est"] * g[col_qty_s]).sum() / g[col_qty_s].sum() if g[col_qty_s].sum() > 0 else np.nan,
        "product_name": g[col_product_name].mode().iloc[0] if g[col_product_name].notna().sum() else np.nan,
        "customer": g[col_customer].mode().iloc[0] if g[col_customer].notna().sum() else np.nan
    }))
    .reset_index(drop=True)
    .rename(columns={col_item_s: col_item})
)

purch_m = purch.assign(_month=purch[col_date_p].dt.to_period("M").dt.to_timestamp()) \
    .groupby([col_item_p, "_month"]).agg(
        qty_p=(col_qty_p, "sum"),
        spend=(col_cost, lambda s: float((s * 0 + 1).rename("ones")))  # dummy to allow next line
    ).reset_index().rename(columns={col_item_p: col_item})
# Correct spend with true line-level computation on original purch df
purch["line_spend"] = purch[col_qty_p] * purch[col_cost]
purch_m = purch.assign(_month=purch[col_date_p].dt.to_period("M").dt.to_timestamp()) \
    .groupby([col_item_p, "_month"]).agg(
        qty_p=(col_qty_p, "sum"),
        spend=("line_spend", "sum"),
        avg_cost=(col_cost, lambda v: weighted_avg(v, purch.loc[v.index, col_qty_p]))
    ).reset_index().rename(columns={col_item_p: col_item})

sales_m = sales.assign(_month=sales[col_date_s].dt.to_period("M").dt.to_timestamp()) \
    .groupby([col_item_s, "_month"]).agg(
        qty_s=(col_qty_s, "sum"),
        revenue=(col_amt, "sum"),
        avg_price=("UnitPrice_Est", lambda v: weighted_avg(v, sales.loc[v.index, col_qty_s])),
        product_name=(col_product_name, lambda x: x.mode().iloc[0] if len(x) and x.notna().sum() else np.nan),
        customer=(col_customer, lambda x: x.mode().iloc[0] if len(x) and x.notna().sum() else np.nan),
    ).reset_index().rename(columns={col_item_s: col_item})

# Item-level weighted average cost (WAC) across filtered window
item_cost_wac = purch.groupby(col_item).apply(lambda df: np.average(df[col_cost], weights=df[col_qty_p]) if df[col_qty_p].sum()>0 else np.nan)
item_price_wavg = sales.groupby(col_item_s).apply(lambda df: np.average(df["UnitPrice_Est"], weights=df[col_qty_s]) if df[col_qty_s].sum()>0 else np.nan)

# Full grid for flows
all_months = pd.period_range(
    start=min(purch_m["_month"].min(), sales_m["_month"].min()),
    end=max(purch_m["_month"].max(), sales_m["_month"].max()),
    freq="M"
).to_timestamp()

all_items = pd.Index(sorted(set(purch_m[col_item]).union(set(sales_m[col_item]))))
flow = (pd.MultiIndex.from_product([all_items, all_months], names=[col_item, "_month"])
        .to_frame(index=False))

flow = flow.merge(purch_m[[col_item, "_month", "qty_p"]], on=[col_item, "_month"], how="left")
flow = flow.merge(sales_m[[col_item, "_month", "qty_s", "revenue"]], on=[col_item, "_month"], how="left")
flow[["qty_p", "qty_s"]] = flow[["qty_p", "qty_s"]].fillna(0.0)
flow = flow.sort_values([col_item, "_month"]).reset_index(drop=True)

# Opening stock map
if opening is not None and "OpeningQty" in opening.columns and ("Item Number" in opening.columns or col_item in opening.columns):
    key = "Item Number" if "Item Number" in opening.columns else col_item
    open_map = opening.set_index(key)["OpeningQty"].to_dict()
else:
    open_map = {}

flow["opening"] = flow[col_item].map(lambda x: open_map.get(x, default_opening))

def compute_running_balance(df):
    cur = None
    bal = []
    for _, r in df.iterrows():
        if cur is None:
            cur = r["opening"] + r.get("qty_p", 0.0) - r.get("qty_s", 0.0)
        else:
            cur = cur + r.get("qty_p", 0.0) - r.get("qty_s", 0.0)
        bal.append(cur)
    return pd.Series(bal, index=df.index)

flow["stock"] = flow.groupby(col_item, group_keys=False).apply(compute_running_balance)

# KPIs
total_revenue = sales[col_amt].sum()
total_qty_sold = sales[col_qty_s].sum()
unique_items = len(all_items)
avg_monthly_sales_qty = sales_m.groupby(col_item)["qty_s"].sum().sum() / max(1, len(all_months))

# Profitability per item (gross margin using WAC)
item_sales_qty = sales.groupby(col_item_s)[col_qty_s].sum()
item_revenue = sales.groupby(col_item_s)[col_amt].sum()
item_cost = item_cost_wac.reindex(item_sales_qty.index) * item_sales_qty  # WAC × qty sold
gross_margin = item_revenue - item_cost
gross_margin_pct = (gross_margin / item_revenue.replace(0, np.nan)) * 100.0
profit_df = pd.DataFrame({
    "Qty Sold": item_sales_qty,
    "Revenue": item_revenue,
    "WAC Cost": item_cost,
    "Gross Margin": gross_margin,
    "GM %": gross_margin_pct
}).sort_values("Gross Margin", ascending=False)

# True vendor spend
purch["LineSpend"] = purch[col_qty_p] * purch[col_cost]
vendor_spend_true = purch.groupby(col_vendor)["LineSpend"].sum().sort_values(ascending=False)

# Days-on-hand (DOH) per SKU at end of range
latest_month = all_months.max()
end_stock = flow[flow["_month"] == latest_month].set_index(col_item)["stock"]
avg_daily_demand = (sales_m.groupby(col_item)["qty_s"].sum() / max(1, len(all_months))) / 30.0
doh = end_stock.reindex(all_items).fillna(0.0) / avg_daily_demand.reindex(all_items).replace(0, np.nan)
doh_df = pd.DataFrame({"Est_Stock_End": end_stock, "AvgDailyDemand": avg_daily_demand, "DaysOnHand": doh}).sort_values("DaysOnHand", ascending=False)

# ------------------------------
# Layout
# ------------------------------
st.title("PAL Radiators — Inventory Intelligence Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue (filtered)", f"{total_revenue:,.0f}")
k2.metric("Total Qty Sold (filtered)", f"{total_qty_sold:,.0f}")
k3.metric("Unique SKUs", f"{unique_items}")
k4.metric("Avg. Monthly Sales (Qty)", f"{avg_monthly_sales_qty:,.1f}")

st.markdown("---")

tab_overview, tab_abc, tab_items, tab_profit_ar, tab_customers, tab_vendors, tab_forecast, tab_replen, tab_downloads = st.tabs(
    ["Overview", "ABC Analysis", "Item Drilldown", "Profitability & AR", "Customers", "Vendors", "Forecast", "Replenishment", "Downloads"]
)

# Overview
with tab_overview:
    st.subheader("Seasonality (Qty Sold by Month)")
    qty_by_item = sales_m.groupby(col_item)["qty_s"].sum().sort_values(ascending=False)
    if len(qty_by_item) == 0:
        st.warning("No sales in the selected period.")
    else:
        top_n = st.slider("Top N SKUs for heatmap", min_value=5, max_value=min(50, len(qty_by_item)), value=min(20, len(qty_by_item)))
        top_items = qty_by_item.head(top_n).index
        pivot = sales_m[sales_m[col_item].isin(top_items)].pivot_table(index=col_item, columns="_month", values="qty_s", aggfunc="sum").fillna(0.0)
        fig = plt.figure(figsize=(10, 0.4*len(top_items) + 2))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(ticks=np.arange(pivot.shape[1]), labels=[d.strftime("%Y-%m") for d in pivot.columns], rotation=90)
        plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)
        plt.title("Sales Quantity Heatmap")
        plt.colorbar()
        st.pyplot(fig, clear_figure=True)

    st.subheader("Aggregate Stock vs. Sales & Purchases")
    agg = flow.groupby("_month").agg(total_stock=("stock", "sum"),
                                     total_sales=("qty_s", "sum"),
                                     total_purch=("qty_p", "sum")).reset_index()
    fig2 = plt.figure(figsize=(10,4))
    plt.plot(agg["_month"], agg["total_stock"], label="Stock (est.)")
    plt.plot(agg["_month"], agg["total_sales"], label="Sales qty")
    plt.plot(agg["_month"], agg["total_purch"], label="Purchase qty")
    plt.legend()
    plt.title("Aggregate Stock & Flows")
    plt.xlabel("Month")
    plt.ylabel("Qty")
    st.pyplot(fig2, clear_figure=True)

# ABC
with tab_abc:
    st.subheader("ABC by Revenue")
    revenue_by_item = sales_m.groupby(col_item)["revenue"].sum().sort_values(ascending=False)
    df_abc = pd.DataFrame({"Revenue": revenue_by_item, "ABC": abc_classification(revenue_by_item)})
    st.dataframe(df_abc)

    st.subheader("ABC by Quantity")
    qty_by_item_sorted = sales_m.groupby(col_item)["qty_s"].sum().sort_values(ascending=False)
    df_abc_q = pd.DataFrame({"Qty": qty_by_item_sorted, "ABC": abc_classification(qty_by_item_sorted)})
    st.dataframe(df_abc_q)

# Item drilldown
with tab_items:
    st.subheader("Per-Item Performance")
    item_sel = st.selectbox("Select Item", sorted(all_items))
    item_flow = flow[flow[col_item] == item_sel].copy()
    item_sales = sales_m[sales_m[col_item] == item_sel].copy()
    item_purch = purch_m[purch_m[col_item] == item_sel].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Sold (qty)", f"{item_sales['qty_s'].sum():,.0f}")
    c2.metric("Revenue", f"{item_sales['revenue'].sum():,.0f}")
    last_stock = item_flow["stock"].iloc[-1] if len(item_flow) else 0.0
    c3.metric("Est. Current Stock", f"{last_stock:,.0f}")

    fig3 = plt.figure(figsize=(10,4))
    plt.plot(item_flow["_month"], item_flow["stock"], label="Stock (est.)")
    plt.plot(item_flow["_month"], item_flow["qty_s"], label="Sales qty")
    plt.plot(item_flow["_month"], item_flow["qty_p"], label="Purch qty")
    plt.legend()
    plt.title(f"Flows — {item_sel}")
    plt.xlabel("Month")
    plt.ylabel("Qty")
    st.pyplot(fig3, clear_figure=True)

    if "avg_price" in sales_m.columns:
        series = item_sales.set_index("_month")["avg_price"].dropna()
        if not series.empty:
            fig4 = plt.figure(figsize=(10,3))
            plt.plot(series.index, series.values)
            plt.title("Average Sales Price Over Time")
            plt.xlabel("Month")
            plt.ylabel("Unit Price (est.)")
            st.pyplot(fig4, clear_figure=True)

# Profitability & AR
with tab_profit_ar:
    st.subheader("Gross Margin by Item (WAC basis)")
    st.dataframe(profit_df)

    figp = plt.figure(figsize=(10,4))
    topg = profit_df.head(min(20, len(profit_df)))
    plt.bar(np.arange(len(topg)), topg["Gross Margin"].values)
    plt.xticks(np.arange(len(topg)), topg.index, rotation=90)
    plt.title("Top Items by Gross Margin")
    plt.ylabel("Gross Margin")
    st.pyplot(figp, clear_figure=True)

    st.subheader("Days On Hand (DOH) by SKU")
    st.dataframe(doh_df)

    # AR / Credit aging
    st.subheader("Credit Aging by Customer")
    # Build invoice-level sales
    inv_df = sales[[col_customer, col_invoice_s, col_amt, col_date_s]].copy()
    inv_df = inv_df.dropna(subset=[col_invoice_s])
    inv_df = inv_df.groupby([col_customer, col_invoice_s, col_date_s], as_index=False)[col_amt].sum()
    today = pd.to_datetime(end_date).normalize()

    if payments is not None:
        # Net outstanding per invoice = sales - payments
        pay_g = payments.groupby(["Customer Name", "Invoice"], as_index=False)["Amount"].sum().rename(columns={"Amount": "Paid"})
        aging = inv_df.merge(pay_g, left_on=[col_customer, col_invoice_s], right_on=["Customer Name", "Invoice"], how="left")
        aging["Paid"] = aging["Paid"].fillna(0.0)
        aging["Outstanding"] = aging[col_amt] - aging["Paid"]
    else:
        # Proxy: treat invoice amount as outstanding (no payments info)
        aging = inv_df.copy()
        aging["Outstanding"] = aging[col_amt]

    aging["Days"] = (today - aging[col_date_s]).dt.days
    bins = [0, 30, 60, 90, 120, 99999]
    labels = ["0-30", "31-60", "61-90", "91-120", "120+"]
    aging["Bucket"] = pd.cut(aging["Days"], bins=bins, labels=labels, right=True, include_lowest=True)

    by_cust_bucket = aging.groupby([col_customer, "Bucket"])["Outstanding"].sum().unstack(fill_value=0.0)
    st.dataframe(by_cust_bucket)

# Customers
with tab_customers:
    st.subheader("Top Customers by Revenue")
    by_cust = sales.groupby(col_customer)[col_amt].sum().sort_values(ascending=False)
    st.dataframe(by_cust.rename("Revenue"))
    fig5 = plt.figure(figsize=(10,4))
    n = min(20, len(by_cust))
    plt.bar(np.arange(n), by_cust.head(n).values)
    plt.xticks(np.arange(n), by_cust.head(n).index, rotation=90)
    plt.title("Top 20 Customers by Revenue")
    plt.ylabel("Revenue")
    st.pyplot(fig5, clear_figure=True)

# Vendors
with tab_vendors:
    st.subheader("True Vendor Spend (qty × unit cost)")
    st.dataframe(vendor_spend_true.rename("Spend"))

    fig6 = plt.figure(figsize=(10,4))
    n = min(20, len(vendor_spend_true))
    plt.bar(np.arange(n), vendor_spend_true.head(n).values)
    plt.xticks(np.arange(n), vendor_spend_true.head(n).index, rotation=90)
    plt.title("Top 20 Vendors by Spend")
    plt.ylabel("Spend")
    st.pyplot(fig6, clear_figure=True)

# Forecast
with tab_forecast:
    st.subheader("Simple Demand Forecast")
    item_for_fc = st.selectbox("Select Item for Forecast", sorted(all_items), key="fc_item")
    ma_window = st.slider("Moving Average window (months)", min_value=1, max_value=12, value=3)
    alpha = st.slider("Exponential smoothing alpha", min_value=0.05, max_value=0.95, value=0.3, step=0.05)

    series_qty = sales_m[sales_m[col_item] == item_for_fc].set_index("_month")["qty_s"].astype(float).sort_index()
    series_qty = series_qty.asfreq("MS").fillna(0.0)

    ma = series_qty.rolling(window=ma_window, min_periods=1).mean()
    es = pd.Series(simple_exp_smooth(series_qty.values, alpha=alpha), index=series_qty.index)

    last_ma = ma.iloc[-1] if len(ma) else 0.0
    last_es = es.iloc[-1] if len(es) else 0.0
    fc_value = (last_ma + last_es) / 2.0

    c1, c2 = st.columns(2)
    c1.metric("Mean monthly demand", f"{series_qty.mean():.2f}")
    c2.metric("Forecast (monthly)", f"{fc_value:.2f}")

    fig7 = plt.figure(figsize=(10,4))
    plt.plot(series_qty.index, series_qty.values, label="Qty/month")
    plt.plot(ma.index, ma.values, label=f"MA({ma_window})")
    plt.plot(es.index, es.values, label=f"ES(alpha={alpha})")
    plt.legend()
    plt.title(f"Demand & Smoothed Series — {item_for_fc}")
    plt.xlabel("Month")
    plt.ylabel("Qty")
    st.pyplot(fig7, clear_figure=True)

# Replenishment
with tab_replen:
    st.subheader("Replenishment Report (per-SKU)")
    # Per-SKU params: lead time, min/max from upload; else defaults
    lt_map = {}
    min_map = {}
    max_map = {}
    if ltmm is not None:
        if "Item Number" in ltmm.columns:
            if "LeadTimeDays" in ltmm.columns:
                lt_map = ltmm.set_index("Item Number")["LeadTimeDays"].to_dict()
            if "MinLevel" in ltmm.columns:
                min_map = ltmm.set_index("Item Number")["MinLevel"].to_dict()
            if "MaxLevel" in ltmm.columns:
                max_map = ltmm.set_index("Item Number")["MaxLevel"].to_dict()

    # Demand stats per SKU
    sales_monthly = sales_m.pivot_table(index=col_item, columns="_month", values="qty_s", aggfunc="sum").fillna(0.0)
    mean_month = sales_monthly.mean(axis=1)
    std_month = sales_monthly.std(axis=1, ddof=1).fillna(0.0)

    # Latest stock per SKU
    latest_stock = flow.groupby(col_item).tail(1).set_index(col_item)["stock"]

    rows = []
    for sku in all_items:
        lt = lt_map.get(sku, lead_time_days_default)
        mn = float(mean_month.get(sku, 0.0))
        sd = float(std_month.get(sku, 0.0))
        demand_lt = mn * (lt / 30.0)
        sigma_lt = sd * math.sqrt(lt / 30.0)
        safety = service_level_z * sigma_lt
        rop = demand_lt + safety
        stock = float(latest_stock.get(sku, 0.0))
        mn_level = float(min_map.get(sku, 0.0))
        mx_level = float(max_map.get(sku, max(stock, rop) + mn))  # fallback max to something sensible

        need = 0.0
        reason = ""
        if stock < mn_level:
            need = max(mx_level - stock, 0.0)
            reason = "Below Min"
        elif stock < rop:
            need = max(rop - stock, 0.0)
            reason = "Below ROP"
        else:
            reason = "OK"

        rows.append({
            "Item": sku,
            "LeadTimeDays": lt,
            "MeanMonthly": mn,
            "StdMonthly": sd,
            "ROP": rop,
            "MinLevel": mn_level,
            "MaxLevel": mx_level,
            "Stock": stock,
            "RecommendedOrder": round(need, 1),
            "Status": reason
        })

    repl_df = pd.DataFrame(rows).set_index("Item").sort_values(["Status", "RecommendedOrder"], ascending=[True, False])
    st.dataframe(repl_df)

# Downloads
with tab_downloads:
    st.subheader("Export Data")
    # Profitability export
    buf_profit = io.StringIO()
    profit_df.to_csv(buf_profit)
    st.download_button("Download Gross Margin by Item (CSV)", data=buf_profit.getvalue(), file_name="gross_margin_by_item.csv")

    # Vendor spend export
    buf_vs = io.StringIO()
    vendor_spend_true.to_csv(buf_vs)
    st.download_button("Download True Vendor Spend (CSV)", data=buf_vs.getvalue(), file_name="vendor_spend_true.csv")

    # DOH export
    buf_doh = io.StringIO()
    doh_df.to_csv(buf_doh)
    st.download_button("Download Days-On-Hand by SKU (CSV)", data=buf_doh.getvalue(), file_name="days_on_hand_by_sku.csv")

    # Replenishment export
    buf_repl = io.StringIO()
    repl_df.to_csv(buf_repl)
    st.download_button("Download Replenishment Report (CSV)", data=buf_repl.getvalue(), file_name="replenishment_report.csv")

st.caption("Built for PAL Radiators | Optional uploads: Opening Stock, Per-SKU LeadTime/Min/Max, Payments for AR aging.")
