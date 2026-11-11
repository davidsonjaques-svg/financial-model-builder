"""Financial Model Builder - Operating Entities (BPL FinmodelSAP logic) as a Streamlit app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import statements as stm
from engine import (ASSET_CLASSES, N_DEPTS, OVERHEAD_SLOTS, blank_deal, build_drivers_from_history,
                    compute, month_index, month_labels, normalise, pmt)

NAVY, GOLD = "#0A1628", "#C9A84C"
SAMPLE = os.path.join(os.path.dirname(__file__), "sample_deal_magand.json")

st.set_page_config(page_title="Financial Model Builder", page_icon="📈", layout="wide")
st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; max-width: 1500px;}}
  .fm-head {{background:{NAVY}; padding:18px 26px; border-radius:6px; border-bottom:4px solid {GOLD};
             margin-bottom:14px;}}
  .fm-head h1 {{color:#fff; font-size:1.55rem; margin:0; letter-spacing:.04em;}}
  .fm-head p {{color:{GOLD}; margin:2px 0 0 0; font-size:.9rem;}}
  div[data-testid="stMetric"] {{background:#F7F5EE; border-left:4px solid {GOLD}; padding:10px 14px;
                                border-radius:4px;}}
  div[data-testid="stMetricValue"] {{font-size:1.35rem; white-space:normal; overflow-wrap:anywhere;}}
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{border-bottom-color:{GOLD}; color:{NAVY};}}
  h3 {{color:{NAVY};}}
</style>""", unsafe_allow_html=True)

S = st.session_state
if "deal" not in S:
    S.deal, S.ver, S.msgs = blank_deal(), 0, []


def K(name: str) -> str:
    """Widget key namespaced by the deal version, so loading a deal refreshes every widget."""
    return f"v{S.ver}:{name}"


def replace_deal(deal: dict, msg: str | None = None):
    S.deal = normalise(deal)
    S.ver += 1
    if msg:
        S.msgs.append(msg)


def flag_refresh():
    S["_refresh"] = True


D = S.deal


# ------------------------------------------------------------------ small input helpers
def money(label, obj, field, name, **kw):
    obj[field] = st.number_input(label, value=float(obj[field] or 0), step=1000.0, format="%.2f",
                                 key=K(name), **kw)
    return obj[field]


def pct(label, obj, field, name, step=0.25, **kw):
    v = st.number_input(label + " (%)", value=round(float(obj[field] or 0) * 100, 6), step=step,
                        format="%.2f", key=K(name), **kw)
    obj[field] = v / 100
    return obj[field]


def integer(label, obj, field, name, **kw):
    obj[field] = int(st.number_input(label, value=int(obj[field] or 0), step=1, key=K(name), **kw))
    return obj[field]


def date(label, obj, field, name, **kw):
    cur = dt.date.fromisoformat(str(obj[field])[:10]) if obj.get(field) else dt.date.today()
    obj[field] = st.date_input(label, value=cur, key=K(name), format="DD/MM/YYYY", **kw).isoformat()
    return obj[field]


def month_grid(name, row_labels, rows, year, as_pct=False, height=None):
    """Editable 12-month grid for one projection year; writes edits straight back into `rows`."""
    labels = month_labels(D["general"]["fye"])[year * 12:(year + 1) * 12]
    base_key = K(f"{name}:base{year}")
    if base_key not in S:
        scale = 100.0 if as_pct else 1.0
        df = pd.DataFrame([[v * scale for v in r[year * 12:(year + 1) * 12]] for r in rows],
                          columns=[f"m{i}" for i in range(12)])
        df.insert(0, "Line", row_labels)
        S[base_key] = df
    cfg = {"Line": st.column_config.TextColumn("Line", disabled=True, width="medium", pinned=True)}
    for i, lab in enumerate(labels):
        cfg[f"m{i}"] = st.column_config.NumberColumn(lab, format="%.2f" if as_pct else "accounting",
                                                     width=80 if as_pct else 112)
    ed = st.data_editor(S[base_key], key=K(f"{name}:ed{year}"), column_config=cfg, hide_index=True,
                        num_rows="fixed", width="stretch",
                        height=height or 38 * (len(rows) + 1) + 4)
    vals = ed[[f"m{i}" for i in range(12)]].fillna(0).to_numpy(dtype=float)
    scale = 0.01 if as_pct else 1.0
    for i, r in enumerate(rows):
        r[year * 12:(year + 1) * 12] = [float(x) * scale for x in vals[i]]


def year_tabs(fn):
    for y, tab in enumerate(st.tabs(["Year 1", "Year 2", "Year 3"])):
        with tab:
            fn(y)


def show(df, decimals=0, height=None):
    """Read-only statement table with headings / totals emphasised."""
    marks = [str(i) for i in df.index]
    clean = stm.clean_index(df)
    labels = [str(i) for i in clean.index]
    small = [r for r, lab in enumerate(labels) if decimals or stm.is_ratio_row(lab)]
    big = [r for r in range(len(labels)) if r not in small]
    BLANK = -9.87654321e-300          # sentinel so empty cells render blank instead of "None"
    num = clean.reset_index(drop=True).astype(float).fillna(BLANK)
    num.columns = [str(c) for c in num.columns]
    cols = list(num.columns)
    num.insert(0, "Line", labels)

    def f0(v):
        return "" if v == BLANK else (f"({abs(v):,.0f})" if v < -0.5 else f"{abs(v):,.0f}")

    def f2(v):
        return "" if v == BLANK else (f"({abs(v):,.2f})" if v < 0 else f"{v:,.2f}")

    def style_row(row):
        m = marks[row.name]
        if m.startswith(stm.TOTAL):
            return ["font-weight:700; background-color:#F4EEDC"] * len(row)
        if m.startswith(stm.HEAD):
            return [f"font-weight:700; color:{NAVY}"] * len(row)
        return [""] * len(row)
    sty = num.style.apply(style_row, axis=1)
    if big:
        sty = sty.format(f0, subset=pd.IndexSlice[big, cols])
    if small:
        sty = sty.format(f2, subset=pd.IndexSlice[small, cols])
    h = height or min(35 * (len(num) + 1) + 4, 900)
    cfg = {"Line": st.column_config.TextColumn(" ", width="medium", pinned=True)}
    if len(cols) > 7:                 # monthly tables: fixed column width, scroll sideways
        cfg.update({c: st.column_config.Column(width=98) for c in cols})
    st.dataframe(sty, width="stretch", height=h, hide_index=True, column_config=cfg)


# ------------------------------------------------------------------ header + sidebar
st.markdown("""<div class="fm-head"><h1>FINANCIAL MODEL BUILDER</h1>
<p>Operating entities &nbsp;|&nbsp; 36-month cash flow, income statement, balance sheet, loans &amp; returns</p>
</div>""", unsafe_allow_html=True)
for m in S.msgs:
    st.toast(m)
S.msgs = []
summary = st.container()

with st.sidebar:
    st.markdown("### Deal file")
    st.caption("Everything you capture lives in one deal file. Download it to save your work; "
               "upload it to carry on later.")
    up = st.file_uploader("Open a saved deal (.json)", type=["json"], key=K("up_json"))
    if up is not None and st.button("Load deal file", width="stretch"):
        try:
            replace_deal(json.load(up), "Deal file loaded")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {e}")
    xl = st.file_uploader("Import an existing FinmodelSAP workbook (.xlsm / .xlsx)", type=["xlsm", "xlsx"],
                          key=K("up_xl"))
    if xl is not None and st.button("Import workbook", width="stretch"):
        try:
            from importer import import_finmodel
            imp = import_finmodel(xl)
            warns = imp.pop("_import_warnings", [])
            replace_deal(imp, "Workbook imported")
            S["import_warnings"] = warns
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Import failed: {e}")
    c1, c2 = st.columns(2)
    if c1.button("New blank deal", width="stretch"):
        replace_deal(blank_deal(), "Started a blank deal")
        S["import_warnings"] = []
        st.rerun()
    if os.path.exists(SAMPLE) and c2.button("Load sample", width="stretch"):
        replace_deal(json.load(open(SAMPLE)), "Sample deal loaded")
        st.rerun()
    for w in S.get("import_warnings", []):
        st.warning(w)
    downloads = st.container()

tabs = st.tabs(["1 · Deal setup", "2 · Financial history", "3 · Trading forecast", "4 · Funding & loans",
                "5 · Assets & other", "Results", "Checks"])

# ================================================================== 1. SETUP
with tabs[0]:
    G, A = D["general"], D["assumptions"]
    c1, c2, c3 = st.columns([2, 1, 1])
    G["applicant"] = c1.text_input("Applicant name", G["applicant"], key=K("applicant"))
    G["deal_type"] = c2.selectbox("Deal type", ["Loan Partner", "Risk Partner", "Equity Partner", "Property"],
                                  index=["Loan Partner", "Risk Partner", "Equity Partner", "Property"].index(
                                      G["deal_type"]) if G["deal_type"] in
                                  ["Loan Partner", "Risk Partner", "Equity Partner", "Property"] else 0,
                                  key=K("deal_type"))
    with c3:
        date("Next financial year end", G, "fye", "fye",
             help="The projection runs for the 36 months starting the month after the previous year end.")
    lab = month_labels(G["fye"])
    st.caption(f"Projection period: **{lab[0]} to {lab[-1]}** (month 1 = {lab[0]}).")
    G["prepared_by"] = st.text_input("Prepared by", G.get("prepared_by", ""), key=K("prep"))

    st.markdown("### Assumptions")
    c = st.columns(4)
    with c[0]:
        pct("Company tax rate", A, "tax_rate", "tax_rate")
        pct("Prime lending rate", A, "prime_rate", "prime")
        pct("Base lending rate (for contribution)", A, "base_rate", "base")
    with c[1]:
        pct("Overdraft interest rate", A, "overdraft_rate", "odr")
        money("Approved overdraft facility (R)", A, "overdraft_limit", "odl")
        pct("Admin & due diligence fee on total loan", A, "admin_fee_pct", "fee")
    with c[2]:
        pct("VAT rate on operations", A, "vat_ops", "vat_ops",
            help="Leave at 0 to model everything excluding VAT (the usual template setting).")
        pct("VAT rate on capital expenditure & royalties", A, "vat_capex", "vat_capex")
        pct("Discount rate for NPV", A, "npv_rate", "npv")
    with c[3]:
        pct("Expected IRR", A, "target_irr", "tirr")
        pct("Sales growth after year 3", A, "sales_growth_after_y3", "sg3")
        pct("Profit growth after year 3", A, "profit_growth_after_y3", "pg3")

    st.markdown("### Debtor & creditor terms")
    st.caption("Share of each month's sales / purchases settled in cash and after 30-120 days. "
               "Each column must add up to 100%.")
    tk = K("terms_base")
    if tk not in S:
        S[tk] = pd.DataFrame({"Term": ["Cash", "30 days", "60 days", "90 days", "120 days"],
                              "Sales %": [x * 100 for x in D["terms"]["debtors"]],
                              "Purchases %": [x * 100 for x in D["terms"]["creditors"]]})
    te = st.data_editor(S[tk], key=K("terms_ed"), hide_index=True, num_rows="fixed",
                        column_config={"Term": st.column_config.TextColumn(disabled=True)}, width=420)
    D["terms"]["debtors"] = [float(x) / 100 for x in te["Sales %"].fillna(0)]
    D["terms"]["creditors"] = [float(x) / 100 for x in te["Purchases %"].fillna(0)]

    st.markdown("### Royalty fee")
    st.caption("Royalty as a % of monthly turnover, charged from the month before the first loan instalment "
               "and paid one month in arrears. Up to three rate tiers; each applies up to and including its "
               "'until month' (month 1 = first projected month).")
    c = st.columns(6)
    for i, tier in enumerate(D["royalty"]["tiers"]):
        with c[i * 2]:
            pct(f"Tier {i+1} royalty", tier, "rate", f"roy_rate{i}", step=0.01)
        with c[i * 2 + 1]:
            integer(f"Tier {i+1} until month", tier, "until_month", f"roy_until{i}", min_value=0)

    st.markdown("### Company tax payments")
    st.caption("Months of each financial year in which tax is paid, and the share of that year's tax paid in each.")
    c = st.columns(3)
    for k in range(3):
        D["tax"]["periods"][k] = int(c[k].number_input(f"Payment period {k+1} (month of year)", 1, 12,
                                                       int(D["tax"]["periods"][k]), key=K(f"taxp{k}")))
    xk = K("tax_base")
    if xk not in S:
        S[xk] = pd.DataFrame([[v * 100 for v in row] for row in D["tax"]["split"]],
                             columns=["Period 1 %", "Period 2 %", "Period 3 %"],
                             index=["Year 1", "Year 2", "Year 3"])
    xe = st.data_editor(S[xk], key=K("tax_ed"), num_rows="fixed", width=520)
    D["tax"]["split"] = [[float(v) / 100 for v in row] for row in xe.fillna(0).to_numpy()]
    D["tax"]["settle_opening_liability"] = st.checkbox(
        "Settle the opening Receiver of Revenue balance together with year 1's tax",
        value=bool(D["tax"].get("settle_opening_liability", True)), key=K("tax_open"))

# ================================================================== 2. HISTORY
with tabs[1]:
    H = D["history"]
    ycols = [f"Actual {str(y)[:7]}" for y in H["years"]]
    st.caption("Capture the last three sets of annual financial statements. The most recent year is the "
               "opening position for the projection.")
    c = st.columns(3)
    for i in range(3):
        with c[i]:
            H["years"][i] = st.date_input(f"Year end {i+1}", dt.date.fromisoformat(str(H["years"][i])[:10]),
                                          key=K(f"hy{i}"), format="DD/MM/YYYY").isoformat()

    st.markdown("### Income statement")
    is_rows = [("sales", "Sales (excl VAT)"), ("opening_stock", "Opening stock"), ("purchases", "Purchases"),
               ("closing_stock", "Closing stock"), ("other_income", "Other income"),
               ("depreciation", "Depreciation"), ("interest_bp", "Interest - Business Partners"),
               ("interest_other", "Interest - other"), ("tax", "Taxation"), ("dividends", "Dividends")]
    ik = K("his_base")
    if ik not in S:
        S[ik] = pd.DataFrame([[lab] + list(H["is"][k]) for k, lab in is_rows], columns=["Line", "y0", "y1", "y2"])
    ycfg = {f"y{i}": st.column_config.NumberColumn(ycols[i], format="localized") for i in range(3)}
    ie = st.data_editor(S[ik], key=K("his_ed"), hide_index=True, num_rows="fixed", width="stretch",
                        column_config={"Line": st.column_config.TextColumn(disabled=True), **ycfg})
    for r, (k, _) in enumerate(is_rows):
        H["is"][k] = [float(x) for x in ie.loc[r, ["y0", "y1", "y2"]].fillna(0)]

    st.markdown("### Overheads")
    st.caption("Rename lines as needed (blank = unused). **Method** drives the 'fill from history' helper on the "
               "Trading tab: 1 = escalate annually, 2 = % of sales, 3 = no increase.")
    ok = K("oh_base")
    if ok not in S:
        S[ok] = pd.DataFrame({"Overhead line": D["overheads"]["names"], "Method": D["overheads"]["method"],
                              "% of sales": [x * 100 for x in D["overheads"]["pct_of_sales"]],
                              "y0": [r[0] for r in H["is"]["overheads"]],
                              "y1": [r[1] for r in H["is"]["overheads"]],
                              "y2": [r[2] for r in H["is"]["overheads"]]})
    oe = st.data_editor(S[ok], key=K("oh_ed"), hide_index=True, num_rows="fixed", width="stretch",
                        on_change=flag_refresh, height=38 * 19 + 4,
                        column_config={"Method": st.column_config.SelectboxColumn(options=[1, 2, 3], required=True),
                                       "% of sales": st.column_config.NumberColumn(format="%.2f"), **ycfg})
    D["overheads"]["names"] = [("" if x is None else str(x)) for x in oe["Overhead line"]]
    D["overheads"]["method"] = [int(x) for x in oe["Method"].fillna(1)]
    D["overheads"]["pct_of_sales"] = [float(x) / 100 for x in oe["% of sales"].fillna(0)]
    H["is"]["overheads"] = [[float(v) for v in row] for row in oe[["y0", "y1", "y2"]].fillna(0).to_numpy()]

    st.markdown("### Balance sheet")
    st.caption("Enter liabilities and short-term portions as positive amounts. Stock comes from the income "
               "statement above; fixed-asset book values for the latest year should agree with the asset "
               "register on tab 5.")
    bs_rows = [("share_capital", "Share capital"), ("ndr", "Non-distributable reserve"),
               ("retained_bf", "Retained income brought forward (excl. this year's profit)"),
               ("sh_loans", "Shareholders' loans"), ("bp_loans", "Business Partners loans (total)"),
               ("bp_loans_st", "   of which short-term portion"), ("other_loans", "Other loans (total)"),
               ("other_loans_st", "   of which short-term portion"), ("fa_land", "Land & buildings (book value)"),
               ("fa_equipment", "Furniture & equipment (book value)"), ("fa_vehicles", "Motor vehicles (book value)"),
               ("goodwill", "Goodwill"), ("investments", "Investments"), ("debtors", "Trade debtors & VAT"),
               ("bank", "Bank (favourable balance)"), ("staff_loans", "Staff loans & deposits"),
               ("tax_advance", "Taxation paid in advance"), ("creditors", "Trade creditors and VAT"),
               ("sundry_creditors", "Sundry creditors"), ("overdraft", "Bank overdraft"),
               ("receiver", "Receiver of Revenue"), ("other_cl", "Other current liabilities")]
    bk = K("hbs_base")
    if bk not in S:
        S[bk] = pd.DataFrame([[lab] + list(H["bs"][k]) for k, lab in bs_rows], columns=["Line", "y0", "y1", "y2"])
    be = st.data_editor(S[bk], key=K("hbs_ed"), hide_index=True, num_rows="fixed", width="stretch",
                        height=38 * 23 + 4,
                        column_config={"Line": st.column_config.TextColumn(disabled=True), **ycfg})
    for r, (k, _) in enumerate(bs_rows):
        H["bs"][k] = [float(x) for x in be.loc[r, ["y0", "y1", "y2"]].fillna(0)]

# ================================================================== 3. TRADING
with tabs[2]:
    DP = D["departments"]
    st.markdown("### Departments / income streams")
    c = st.columns(N_DEPTS)
    for i in range(N_DEPTS):
        DP["names"][i] = c[i].text_input(f"Department {i+1}", DP["names"][i], key=K(f"dn{i}"),
                                         on_change=flag_refresh)
    dnames = [n or f"(unused {i+1})" for i, n in enumerate(DP["names"])]

    with st.expander("Fill the monthly grids from history (optional starting point)"):
        st.caption("Projects flat monthly sales off last year's turnover and overheads off last year's actuals "
                   "(using each line's method). This **overwrites** the sales, GP % and overhead grids below - "
                   "refine month by month afterwards.")
        c = st.columns(6)
        g = [c[i].number_input(f"Sales growth year {i+1} (%)", value=10.0, step=1.0, key=K(f"hg{i}")) / 100
             for i in range(3)]
        for i in range(3):
            D["overheads"]["escalation"][i] = c[3 + i].number_input(
                f"Overhead escalation year {i+1} (%)", value=float(D["overheads"]["escalation"][i]) * 100,
                step=0.5, key=K(f"he{i}")) / 100
        c = st.columns(N_DEPTS)
        split = [c[i].number_input(f"{dnames[i]} - share of sales (%)", value=100.0 if i == 0 else 0.0,
                                   step=5.0, key=K(f"hs{i}")) / 100 for i in range(N_DEPTS)]
        hi = stm_hist = None
        last_sales = D["history"]["is"]["sales"][2]
        last_cos = (D["history"]["is"]["opening_stock"][2] + D["history"]["is"]["purchases"][2]
                    - D["history"]["is"]["closing_stock"][2])
        hgp = (last_sales - last_cos) / last_sales * 100 if last_sales else 0.0
        c = st.columns(N_DEPTS)
        gps = [c[i].number_input(f"{dnames[i]} - GP (%)", value=round(hgp, 2), step=0.5, key=K(f"hgp{i}")) / 100
               for i in range(N_DEPTS)]
        if st.button("Fill grids from history", type="primary"):
            if abs(sum(split) - 1) > 1e-6:
                st.error("Department shares must add up to 100%.")
            else:
                replace_deal(build_drivers_from_history(D, g, split, gps), "Monthly grids filled from history")
                st.rerun()

    st.markdown("### Monthly sales (excl VAT)")
    year_tabs(lambda y: month_grid("sales", dnames, DP["sales"], y))
    st.markdown("### Gross profit % by department")
    year_tabs(lambda y: month_grid("gp", dnames, DP["gp"], y, as_pct=True))
    st.markdown("### Monthly overhead payments")
    st.caption("Depreciation, royalties and the admin & due diligence fee are added automatically.")
    onames = [n or f"(unused {i+1})" for i, n in enumerate(D["overheads"]["names"])]
    year_tabs(lambda y: month_grid("oh", onames, D["overheads"]["monthly"], y, height=38 * 19 + 4))

# ================================================================== 4. FUNDING
with tabs[3]:
    st.markdown("### New Business Partners term loans")
    for i, tab in enumerate(st.tabs(["Long term loan 1", "Long term loan 2", "Long term loan 3"])):
        ln = D["new_bp_loans"][i]
        with tab:
            c = st.columns(4)
            with c[0]:
                money("Capital amount (R)", ln, "capital", f"bp{i}cap")
            with c[1]:
                integer("Term (months)", ln, "term", f"bp{i}term", min_value=0)
            with c[2]:
                pct("Interest rate", ln, "rate", f"bp{i}rate")
            with c[3]:
                ln["auto_instalment"] = st.checkbox("Calculate instalment", value=bool(ln.get("auto_instalment", True)),
                                                    key=K(f"bp{i}auto"))
                if ln["auto_instalment"]:
                    ln["instalment"] = pmt(ln["rate"], int(ln["term"]), ln["capital"])
                    st.caption(f"Instalment: **R {ln['instalment']:,.2f}** per month")
                else:
                    money("Instalment (R)", ln, "instalment", f"bp{i}inst")
            st.caption("Advances / payouts (up to four tranches)")
            c = st.columns(4)
            for j, tr in enumerate(ln["tranches"]):
                with c[j]:
                    date(f"Tranche {j+1} date", tr, "date", f"bp{i}td{j}")
                    money(f"Tranche {j+1} amount (R)", tr, "amount", f"bp{i}ta{j}")
            diff = ln["capital"] - sum(t["amount"] for t in ln["tranches"])
            if abs(diff) > 0.5:
                st.warning(f"Tranches differ from the capital amount by R {diff:,.2f}.")
            c = st.columns(4)
            with c[0]:
                date("First instalment date", ln, "first_instalment", f"bp{i}fi",
                     help="Interest only is paid until this month (moratorium on capital).")
            with c[1]:
                integer("Balloon - model month number (0 = none)", ln, "balloon_period", f"bp{i}bp", min_value=0)
            with c[2]:
                money("Balloon amount (R)", ln, "balloon_amount", f"bp{i}ba")
            if ln["capital"]:
                c[3].caption(f"First payout in model month {month_index(ln['tranches'][0]['date'], D['general']['fye'])}; "
                             f"first instalment in model month {month_index(ln['first_instalment'], D['general']['fye'])}.")

    def existing_block(title, loans, prefix, n):
        st.markdown(f"### {title}")
        c = st.columns(n)
        for i, ln in enumerate(loans):
            with c[i]:
                st.markdown(f"**Loan {i+1}**")
                money("Balance (R)", ln, "balance", f"{prefix}{i}bal")
                money("Instalment (R)", ln, "instalment", f"{prefix}{i}inst")
                pct("Interest rate", ln, "rate", f"{prefix}{i}rate")
                integer("Redeemed in model month (0 = no)", ln, "redeem_period", f"{prefix}{i}rp", min_value=0)
                money("Amount redeemed (R)", ln, "redeem_amount", f"{prefix}{i}ra")

    existing_block("Existing Business Partners loans", D["existing_bp_loans"], "ebp", 2)
    existing_block("Existing other loans", D["existing_other_loans"], "eot", 4)
    st.markdown("### New other / outside loans")
    c = st.columns(2)
    for i, ln in enumerate(D["new_other_loans"]):
        with c[i]:
            st.markdown(f"**Loan {i+1}**")
            money("Capital amount (R)", ln, "capital", f"no{i}cap")
            money("Instalment (R)", ln, "instalment", f"no{i}inst")
            pct("Interest rate", ln, "rate", f"no{i}rate")
            date("Month advanced", ln, "advance_date", f"no{i}ad")
            date("First instalment date", ln, "first_instalment", f"no{i}fi")

# ================================================================== 5. ASSETS & OTHER
with tabs[4]:
    st.markdown("### Fixed asset register - opening position")
    st.caption("Method 1 = straight line on cost, 2 = reducing balance on book value.")
    c = st.columns(4)
    for i, (name, a) in enumerate(zip(ASSET_CLASSES, D["assets"])):
        with c[i]:
            st.markdown(f"**{name}**")
            a["method"] = st.selectbox("Depreciation method", [1, 2], index=0 if a["method"] == 1 else 1,
                                       format_func=lambda v: "1 - Straight line" if v == 1 else "2 - Reducing balance",
                                       key=K(f"as{i}m"))
            pct("Depreciation rate p.a.", a, "rate", f"as{i}r", step=1.0)
            money("Opening cost (R)", a, "opening_cost", f"as{i}c")
            money("Opening book value (R)", a, "opening_book_value", f"as{i}bv")
    st.markdown("### Additions (excl VAT)")
    year_tabs(lambda y: month_grid("adds", ASSET_CLASSES, [a["additions"] for a in D["assets"]], y))

    st.markdown("### Other monthly cash-flow items")
    other = [("stock_increase", "Stock increase (purchases above cost of sales)"),
             ("other_income", "Other income"), ("own_contribution", "Own contribution (to shareholders' loans)"),
             ("staff_loans", "Staff loans & deposits paid"), ("goodwill", "Goodwill paid"),
             ("dividends", "Dividends paid")]
    year_tabs(lambda y: month_grid("other", [lab for _, lab in other], [D["monthly"][k] for k, _ in other], y))

# ================================================================== CALCULATE
res = compute(D)
IS, BS, RT, RA = res["IS"], res["BS"], res["returns"], res["ratios"]
rand = lambda v: "n/a" if v is None else f"R {v:,.0f}"       # noqa: E731


def short(v):
    """Compact rand figure for the headline tiles."""
    if v is None:
        return "n/a"
    a = abs(v)
    txt = f"R {a/1e6:,.2f}m" if a >= 1e6 else (f"R {a/1e3:,.0f}k" if a >= 1e3 else f"R {a:,.0f}")
    return f"({txt})" if v < 0 else txt


pc = lambda v: "n/a" if v is None else f"{v*100:,.2f}%"      # noqa: E731
xx = lambda v: "n/a" if v is None else f"{v:,.2f}x"          # noqa: E731

with summary:
    c = st.columns([2, 1, 1, 1, 1, 1])
    c[0].metric("Applicant", D["general"]["applicant"] or "-")
    c[1].metric("Year 1 turnover", short(IS["sales"][0]))
    c[2].metric("Year 1 NPBT", short(IS["npbt"][0]))
    c[3].metric("Peak overdraft", short(max(-float(res["closing_bank"].min()), 0.0) + 0.0))
    c[4].metric("Year 1 DSCR", xx(RA["dscr"][0]))
    c[5].metric("IRR to funder", pc(RT["irr_total"]))

with tabs[5]:
    rt = st.tabs(["Dashboard", "Cash flow", "Departments", "Overheads", "Income statement", "Balance sheet",
                  "Asset register", "Loans", "Risk & return"])
    with rt[0]:
        st.markdown("### Bank balance against the overdraft facility")
        chart = pd.DataFrame({"Closing bank balance": res["closing_bank"],
                              "Overdraft facility": -res["overdraft_limit"]},
                             index=pd.Index(range(1, 37), name="Model month"))
        st.line_chart(chart, color=[NAVY, GOLD], height=300)
        st.caption("Month 1 = " + res["labels"][0] + ". The balance should stay above the facility line.")
        st.markdown("### Key ratios")
        show(stm.ratio_table(res), decimals=2)
    with rt[1]:
        year_tabs(lambda y: show(stm.cashflow_table(res, y)))
    with rt[2]:
        year_tabs(lambda y: show(stm.dept_table(res, y)))
    with rt[3]:
        year_tabs(lambda y: show(stm.overheads_table(res, y)))
    with rt[4]:
        show(stm.income_table(res), height=1500)
    with rt[5]:
        show(stm.balance_table(res), height=1500)
    with rt[6]:
        year_tabs(lambda y: show(stm.asset_table(res, y)))
    with rt[7]:
        L = res["loans"]
        opts = {"New Business Partners loans (combined)": L["bp_new"],
                "Existing Business Partners loans (combined)": L["bp_exist"],
                "Existing other loans (combined)": L["oth_exist"], "New other loans (combined)": L["oth_new"]}
        for i, s_ in enumerate(L["bp_new_each"]):
            opts[f"New Business Partners loan {i+1}"] = s_
        for i, s_ in enumerate(L["oth_exist_each"]):
            opts[f"Existing other loan {i+1}"] = s_
        pick = st.selectbox("Schedule", list(opts), key=K("loan_pick"))
        months = st.slider("Months to show", 12, 144, 60, 12, key=K("loan_months"))
        st.dataframe(stm.loan_table(res, opts[pick], months).style.format("{:,.2f}"),
                     width="stretch", height=600)
    with rt[8]:
        c = st.columns(4)
        c[0].metric("Total new BP capital", rand(RT["total_capital"]))
        c[1].metric("IRR - loan + fee", pc(RT["irr_loan_fee"]))
        c[2].metric("IRR - incl. royalties", pc(RT["irr_total"]))
        c[3].metric("Contribution above base rate", pc(RT["contribution_pct"]))
        c = st.columns(4)
        c[0].metric("Admin & due diligence fee", rand(RT["admin_fee"]))
        c[1].metric("Royalties over the deal", rand(RT["royalties_total"]))
        c[2].metric(f"NPV at {D['assumptions']['npv_rate']*100:.2f}%", rand(RT["npv"]))
        c[3].metric("Contribution (R)", rand(RT["contribution_rand"]))
        if RT["flows"]:
            st.markdown("### Funder cash flow from first payout (period 0)")
            st.bar_chart(pd.DataFrame({"Cash flow": RT["flows"]},
                                      index=pd.Index(range(len(RT["flows"])), name="Deal period")),
                         color=NAVY, height=260)
        else:
            st.info("Capture a new Business Partners loan on tab 4 to see returns.")

# ================================================================== CHECKS
with tabs[6]:
    checks = []

    def chk(ok, text, detail=""):
        checks.append((ok, text, detail))
    worst = max(abs(x) for x in BS["difference"])
    chk(worst < 1, "Projected balance sheets balance", f"largest difference R {worst:,.2f}")
    hb = stm.balance_table(res)
    hd = [abs(x) for x in hb.loc["DIFFERENCE (must be nil)"].iloc[:3]]
    chk(max(hd) < 1, "Historic balance sheets balance", f"differences: {', '.join(f'R {x:,.0f}' for x in hd)}")
    for nm, k in (("Sales", "debtors"), ("Purchases", "creditors")):
        chk(abs(sum(D["terms"][k]) - 1) < 1e-6, f"{nm} terms add up to 100%", f"{sum(D['terms'][k])*100:.1f}%")
    for y in range(3):
        chk(abs(sum(D["tax"]["split"][y]) - 1) < 1e-6, f"Tax payment split for year {y+1} adds up to 100%",
            f"{sum(D['tax']['split'][y])*100:.1f}%")
    eb = sum(x["balance"] for x in D["existing_bp_loans"])
    chk(abs(eb - D["history"]["bs"]["bp_loans"][2]) < 1, "Existing BP loans agree to the last balance sheet",
        f"loans R {eb:,.0f} vs balance sheet R {D['history']['bs']['bp_loans'][2]:,.0f}")
    eo = sum(x["balance"] for x in D["existing_other_loans"])
    chk(abs(eo - D["history"]["bs"]["other_loans"][2]) < 1, "Existing other loans agree to the last balance sheet",
        f"loans R {eo:,.0f} vs balance sheet R {D['history']['bs']['other_loans'][2]:,.0f}")
    fa_reg = sum(a["opening_book_value"] for a in D["assets"])
    fa_bs = sum(D["history"]["bs"][k][2] for k in ("fa_land", "fa_equipment", "fa_vehicles"))
    chk(abs(fa_reg - fa_bs) < 1, "Asset register opening book value agrees to the last balance sheet",
        f"register R {fa_reg:,.0f} vs balance sheet R {fa_bs:,.0f}")
    for i, ln in enumerate(D["new_bp_loans"]):
        if ln["capital"]:
            dd = ln["capital"] - sum(t["amount"] for t in ln["tranches"])
            chk(abs(dd) < 1, f"New BP loan {i+1}: tranches equal the capital amount", f"difference R {dd:,.2f}")
    breach = [res["labels"][m] for m in range(36) if res["closing_bank"][m] < -res["overdraft_limit"] - 1]
    chk(not breach, "Bank balance stays within the overdraft facility",
        "within limit throughout" if not breach else f"exceeded in {len(breach)} month(s): {', '.join(breach[:12])}"
        + ("..." if len(breach) > 12 else ""))
    for y in range(3):
        v = RA["dscr"][y]
        chk(v is None or v >= 1.25, f"Year {y+1} debt service cover of at least 1.25x", xx(v))
    for ok, text, detail in checks:
        (st.success if ok else st.error)(f"{'✔' if ok else '✘'}  {text} - {detail}")
    st.caption("The 1.25x debt-service threshold is a rule of thumb in this app, not a template setting.")

# ================================================================== downloads
with downloads:
    st.markdown("### Save & export")
    safe = "".join(ch for ch in (D["general"]["applicant"] or "deal") if ch.isalnum() or ch in " _-").strip()
    st.download_button("Download deal file (.json)", json.dumps(D, indent=1, default=str),
                       file_name=f"FinModel - {safe}.json", mime="application/json", width="stretch")
    st.download_button("Download Excel pack (.xlsx)", stm.excel_pack(res),
                       file_name=f"Financial Model - {safe}.xlsx", width="stretch", type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("The Excel pack holds values only: income statement, balance sheet, monthly cash flows, "
               "departments, overheads, asset register, loan schedules, ratios and returns.")

if S.pop("_refresh", False):
    S.ver += 1
    st.rerun()
