
PAL Radiators — Inventory Intelligence (Streamlit)

Deploy from GitHub (Streamlit Community Cloud)
----------------------------------------------
1) Push these to your GitHub repo:
   - pal_radiators_app.py
   - (optional) data/PAL Radiator Purchase and Sold - 2022 onwards.xlsx

2) On Streamlit Community Cloud:
   - New app → choose your repo/branch → main file: pal_radiators_app.py

Data Options
------------
Option A — Upload Excel at runtime in the app (must include sheets named 'Purchases' and 'Sales').
Option B — Commit the Excel in your repo (e.g., under `data/`) and enter the relative path in the sidebar.

New: Optional CSV Uploads
-------------------------
1) **Opening Stock CSV**
   Columns: `Item Number, OpeningQty`

2) **Lead Time & Levels CSV**
   Columns: `Item Number, LeadTimeDays, MinLevel, MaxLevel`

3) **Payments CSV** (for true AR aging)
   Columns: `Customer Name, Invoice, Amount, Date`
   - If omitted, the Credit Aging view will bucket invoice amounts as a proxy for A/R (not net of payments).

What's New / Extended
---------------------
- **Gross margin** per item using weighted-average cost (WAC) from purchases vs. weighted-average selling price.
- **True vendor spend** computed from purchase line totals (qty × unit cost), not approximate averages.
- **Credit aging by customer** (0–30, 31–60, 61–90, 91–120, 120+) with optional payments netting.
- **Days-on-hand** (DOH) per SKU = Ending stock / Avg daily demand.
- **Per-SKU lead times & min/max** upload with a **Replenishment Report** (ROP, status, and recommended order quantity).

Run locally (optional)
----------------------
pip install streamlit pandas numpy matplotlib openpyxl
streamlit run pal_radiators_app.py
