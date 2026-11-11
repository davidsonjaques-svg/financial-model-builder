"""Turn engine results into presentation tables (pandas DataFrames) and an Excel pack."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd

from engine import OVERHEAD_SLOTS, ASSET_CLASSES, N_DEPTS

HEAD = "§"   # marker prefix for section-heading rows
TOTAL = "∑"  # marker prefix for total rows


def _ycols(res, y):
    return res["labels"][y * 12:(y + 1) * 12]


def _frame(rows, cols, total=True):
    """rows: list of (label, array-like | None). None -> heading row."""
    data, idx = [], []
    for lab, v in rows:
        idx.append(lab)
        if v is None:
            data.append([np.nan] * (len(cols) + (1 if total else 0)))
        else:
            v = list(np.asarray(v, dtype=float))
            data.append(v + ([float(np.sum(v))] if total else []))
    return pd.DataFrame(data, index=idx, columns=list(cols) + (["Total"] if total else []))


def dept_table(res, y):
    d, sl = res["deal"], slice(y * 12, (y + 1) * 12)
    names = [n or f"Department {i+1}" for i, n in enumerate(d["departments"]["names"])]
    act = [i for i in range(N_DEPTS) if np.any(res["dept"]["sales"][i]) or d["departments"]["names"][i]]
    rows = [(HEAD + "Sales", None)] + [(names[i], res["dept"]["sales"][i][sl]) for i in act]
    rows += [(TOTAL + "Total sales", res["sales"][sl]), (HEAD + "Cost of sales", None)]
    rows += [(names[i] + " ", res["dept"]["cos"][i][sl]) for i in act]
    rows += [(TOTAL + "Total cost of sales", res["cos"][sl]),
             (TOTAL + "Gross profit", res["sales"][sl] - res["cos"][sl])]
    df = _frame(rows, _ycols(res, y))
    gp = (res["sales"][sl] - res["cos"][sl]) / np.where(res["sales"][sl] == 0, np.nan, res["sales"][sl])
    tot = df.loc[TOTAL + "Gross profit", "Total"] / df.loc[TOTAL + "Total sales", "Total"] \
        if df.loc[TOTAL + "Total sales", "Total"] else np.nan
    df.loc["Gross profit %"] = list(np.nan_to_num(gp) * 100) + [tot * 100]
    return df


def cashflow_table(res, y):
    sl = slice(y * 12, (y + 1) * 12)
    rows = [("Sales (excl VAT)", res["sales"][sl]), ("Cost of sales (excl VAT)", res["cos"][sl]),
            ("Stock increase", res["stock_increase"][sl]),
            (HEAD + "CASH RECEIPTS", None)]
    rows += [(k, v[sl]) for k, v in res["receipts"].items()]
    rows += [(TOTAL + "TOTAL RECEIPTS", res["total_receipts"][sl]), (HEAD + "CASH PAYMENTS", None)]
    rows += [(k, v[sl]) for k, v in res["payments"].items()]
    rows += [(TOTAL + "TOTAL PAYMENTS", res["total_payments"][sl]),
             (TOTAL + "SURPLUS (SHORTFALL)", res["total_receipts"][sl] - res["total_payments"][sl])]
    df = _frame(rows, _ycols(res, y))
    ob, cb = res["opening_bank"][sl], res["closing_bank"][sl]
    df.loc["BANK: Opening balance (overdraft)"] = list(ob) + [ob[0]]
    df.loc[TOTAL + "BANK: Closing balance (overdraft)"] = list(cb) + [cb[-1]]
    df.loc["Overdraft facility"] = [res["overdraft_limit"]] * 13
    df.loc["Headroom on facility"] = list(cb + res["overdraft_limit"]) + [cb[-1] + res["overdraft_limit"]]
    return df


def overheads_table(res, y):
    d, sl = res["deal"], slice(y * 12, (y + 1) * 12)
    rows = [(d["overheads"]["names"][i], res["overhead_lines"][i][sl]) for i in range(OVERHEAD_SLOTS)
            if d["overheads"]["names"][i] or np.any(res["overhead_lines"][i])]
    rows += [("Administration & Due Diligence Fee", res["admin_fee"][sl]), ("Royalties", res["royalties"][sl]),
             (TOTAL + "TOTAL OVERHEADS (cash)", res["overheads_cash"][sl]),
             ("Depreciation", res["depreciation"][sl])]
    return _frame(rows, _ycols(res, y))


def _ylabels(res):
    hy = [f"Actual {str(x)[:7]}" for x in res["deal"]["history"]["years"]]
    py = [f"Estimate {d.isoformat()[:7]}" for d in res["year_ends"]]
    return hy + py


def income_table(res):
    d, H, P = res["deal"], res["hist"], res["IS"]
    both = lambda k: list(H[k]) + list(P[k])  # noqa: E731
    rows = [("Sales (excl VAT)", both("sales")), ("Less: Cost of sales", both("cos")),
            ("   Opening stock", both("opening_stock")), ("   Add: Purchases", both("purchases")),
            ("   Less: Closing stock", both("closing_stock")),
            (TOTAL + "Gross profit", both("gross_profit")),
            ("Gross profit %", [x * 100 for x in both("gp_pct")]),
            ("Add: Other income", both("other_income")), (HEAD + "Less: Overheads", None)]
    for i in range(OVERHEAD_SLOTS):
        v = list(H["overhead_lines"][i]) + list(P["overhead_lines"][i])
        if d["overheads"]["names"][i] or any(v):
            rows.append(("   " + (d["overheads"]["names"][i] or f"Overhead {i+1}"), v))
    rows += [("   Royalties", [0, 0, 0] + P["royalties"]),
             ("   Administration & Due Diligence Fee", [0, 0, 0] + P["admin_fee"]),
             ("   Depreciation", both("depreciation")), (TOTAL + "Total overheads", both("overheads")),
             (TOTAL + "NPBIT", both("npbit")),
             ("Less: Interest - Business Partners", both("interest_bp")),
             ("Less: Interest - Other", both("interest_other")), (TOTAL + "NPBT", both("npbt")),
             ("Less: Taxation", both("tax")), (TOTAL + "Net profit (after interest & tax)", both("np_after_tax")),
             ("Less: Dividends", both("dividends")), ("Net profit (after dividends)", both("np_after_div")),
             ("Add: Depreciation", both("depreciation")), (TOTAL + "Funds generated", both("funds_generated")),
             ("Less: Capital portion - Business Partners", [np.nan] * 3 + P["capital_bp"]),
             ("Less: Capital portion - Other", [np.nan] * 3 + P["capital_other"]),
             (TOTAL + "NET FUNDS GENERATED", [np.nan] * 3 + P["net_funds"]),
             (HEAD + "Breakeven analysis", None),
             ("Average turnover / month", both("avg_turnover")),
             ("Breakeven / month - cash", [np.nan] * 3 + P["breakeven_cash"]),
             ("Breakeven / month - profit", [np.nan] * 3 + P["breakeven_profit"])]
    return _frame(rows, _ylabels(res), total=False)


def balance_table(res):
    h, B, H = res["deal"]["history"]["bs"], res["BS"], res["hist"]
    hv = lambda k: list(h[k])  # noqa: E731
    hist = {k: hv(k) for k in h}
    hist["net_income"] = list(H["np_after_div"])
    hist["equity"] = [sum(hist[k][y] for k in ("share_capital", "ndr", "net_income", "retained_bf", "sh_loans"))
                      for y in range(3)]
    hist["lt_loans"] = [hist["bp_loans"][y] - hist["bp_loans_st"][y] + hist["other_loans"][y]
                        - hist["other_loans_st"][y] for y in range(3)]
    hist["capital_employed"] = [hist["equity"][y] + hist["lt_loans"][y] for y in range(3)]
    hist["fixed_assets"] = [hist["fa_land"][y] + hist["fa_equipment"][y] + hist["fa_vehicles"][y] for y in range(3)]
    hist["lt_assets"] = [hist["goodwill"][y] + hist["investments"][y] for y in range(3)]
    hist["stock"] = list(H["closing_stock"])
    hist["current_assets"] = [sum(hist[k][y] for k in ("stock", "debtors", "bank", "staff_loans", "tax_advance"))
                              for y in range(3)]
    hist["st_loans"] = [hist["bp_loans_st"][y] + hist["other_loans_st"][y] for y in range(3)]
    hist["current_liabilities"] = [sum(hist[k][y] for k in ("st_loans", "creditors", "sundry_creditors",
                                                            "overdraft", "receiver", "other_cl")) for y in range(3)]
    hist["net_current_assets"] = [hist["current_assets"][y] - hist["current_liabilities"][y] for y in range(3)]
    hist["employment_of_capital"] = [hist["fixed_assets"][y] + hist["lt_assets"][y] + hist["net_current_assets"][y]
                                     for y in range(3)]
    hist["difference"] = [hist["employment_of_capital"][y] - hist["capital_employed"][y] for y in range(3)]
    both = lambda k, sign=1: [sign * x for x in hist[k]] + [sign * x for x in B[k]]  # noqa: E731
    rows = [(TOTAL + "CAPITAL EMPLOYED", both("capital_employed")),
            (HEAD + "Shareholders' equity", None), ("   Share capital", both("share_capital")),
            ("   Non-distributable reserve", both("ndr")), ("   Net income - this period", both("net_income")),
            ("   Retained income (b/fwd)", both("retained_bf")), ("   Shareholders' loans", both("sh_loans")),
            (TOTAL + "Total shareholders' equity", both("equity")),
            (HEAD + "Long term loans", None), ("   Business Partners", both("bp_loans")),
            ("   Less: Short term portion", both("bp_loans_st", -1)), ("   Other loans", both("other_loans")),
            ("   Less: Short term portion ", both("other_loans_st", -1)),
            (TOTAL + "Total long term loans", both("lt_loans")),
            (TOTAL + "EMPLOYMENT OF CAPITAL", both("employment_of_capital")),
            (HEAD + "Fixed assets (cost - depreciation)", None), ("   Land & Buildings", both("fa_land")),
            ("   Furniture & Equipment", both("fa_equipment")), ("   Motor Vehicles", both("fa_vehicles")),
            (TOTAL + "Total fixed assets", both("fixed_assets")),
            ("   Goodwill", both("goodwill")), ("   Investments", both("investments")),
            (TOTAL + "Total long term assets", both("lt_assets")),
            (HEAD + "Current assets", None), ("   Stock", both("stock")), ("   Trade debtors & VAT", both("debtors")),
            ("   Bank", both("bank")), ("   Staff loans & deposits", both("staff_loans")),
            ("   Taxation paid in advance", both("tax_advance")),
            (TOTAL + "Total current assets", both("current_assets")),
            (HEAD + "Less: Current liabilities", None), ("   Short term portion of L/T loans", both("st_loans")),
            ("   Trade creditors and VAT", both("creditors")), ("   Sundry creditors", both("sundry_creditors")),
            ("   Bank overdraft", both("overdraft")), ("   Receiver of Revenue", both("receiver")),
            ("   Other", both("other_cl")), (TOTAL + "Total current liabilities", both("current_liabilities")),
            (TOTAL + "NET CURRENT ASSETS", both("net_current_assets")),
            ("DIFFERENCE (must be nil)", both("difference"))]
    return _frame(rows, _ylabels(res), total=False)


def asset_table(res, y):
    sl = slice(y * 12, (y + 1) * 12)
    rows = []
    for name, a in zip(ASSET_CLASSES, res["assets"]):
        rows += [(HEAD + name, None), (f"   Additions (excl VAT) - {name}", a["additions"][sl]),
                 (f"   Depreciation - {name}", a["depreciation"][sl])]
    rows.append((TOTAL + "Total depreciation", res["depreciation"][sl]))
    df = _frame(rows, _ycols(res, y))
    for name, a in zip(ASSET_CLASSES, res["assets"]):
        bv = a["book_value"][sl]
        df.loc[f"   Book value - {name}"] = list(bv) + [bv[-1]]
    order = []
    for name in ASSET_CLASSES:
        order += [HEAD + name, f"   Additions (excl VAT) - {name}", f"   Depreciation - {name}",
                  f"   Book value - {name}"]
    return df.loc[order + [TOTAL + "Total depreciation"]]


def loan_table(res, sched, months=60):
    lab = res["labels"]
    from engine import month_labels
    lab = month_labels(res["deal"]["general"]["fye"], months)
    df = pd.DataFrame({"Month": lab, "Payout": sched["payout"][1:months + 1],
                       "Interest": sched["interest"][1:months + 1],
                       "Capital": (sched["capital"] + sched["redeem"])[1:months + 1],
                       "Payment": (sched["payment"] + sched["redeem"])[1:months + 1],
                       "Balance": sched["balance"][1:months + 1]})
    return df.set_index("Month")


def ratio_table(res):
    R, I = res["ratios"], res["IS"]
    cols = [f"Year {i+1} ({d.isoformat()[:7]})" for i, d in enumerate(res["year_ends"])]
    pct = lambda v: [None if x is None else x * 100 for x in v]  # noqa: E731
    rows = {"Gross profit %": pct(R["gp_pct"]), "Net profit after tax %": pct(R["np_pct"]),
            "Current ratio (x)": R["current_ratio"], "Acid test (x)": R["acid_test"],
            "Gearing: interest-bearing debt / equity (x)": R["gearing"],
            "Interest cover: NPBIT / interest (x)": R["interest_cover"],
            "Debt service cover: (NPBIT + depr - tax) / (interest + capital) (x)": R["dscr"],
            "Debtor days": R["debtor_days"], "Creditor days": R["creditor_days"],
            "Breakeven turnover / month - cash (R)": I["breakeven_cash"],
            "Breakeven turnover / month - profit (R)": I["breakeven_profit"],
            "Average turnover / month (R)": I["avg_turnover"],
            "Peak overdraft in year (R)": R["peak_overdraft"],
            "Lowest headroom on facility (R)": R["min_headroom"]}
    return pd.DataFrame(rows, index=cols).T


def is_ratio_row(label: str) -> bool:
    """Rows shown with two decimals (percentages, multiples, day counts)."""
    return "%" in label or "(x)" in label or label.strip() in ("Debtor days", "Creditor days")


def clean_index(df):
    out = df.copy()
    out.index = [str(i).replace(HEAD, "").replace(TOTAL, "") for i in out.index]
    return out


# --------------------------------------------------------------------------- Excel pack
def excel_pack(res) -> bytes:
    d = res["deal"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book
        f_title = wb.add_format({"bold": True, "font_size": 14, "font_color": "#0A1628"})
        f_sub = wb.add_format({"italic": True, "font_color": "#666666"})
        f_hdr = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#0A1628", "align": "center",
                               "border": 1})
        f_num = wb.add_format({"num_format": "#,##0;[Red](#,##0);-"})
        f_dec = wb.add_format({"num_format": "#,##0.00;[Red](#,##0.00);-"})
        f_tot = wb.add_format({"num_format": "#,##0;[Red](#,##0);-", "bold": True, "top": 1, "bg_color": "#F4EEDC"})
        f_lab = wb.add_format({})
        f_labtot = wb.add_format({"bold": True, "top": 1, "bg_color": "#F4EEDC"})
        f_sec = wb.add_format({"bold": True, "font_color": "#0A1628", "bottom": 2, "bottom_color": "#C9A84C"})

        def put(sheet, title, df, decimals=False):
            ws = wb.add_worksheet(sheet[:31])
            ws.write(0, 0, f"{d['general']['applicant']} - {title}", f_title)
            ws.write(1, 0, "Values only. Generated by the Financial Model Builder (FinmodelSAP operating-entity logic).", f_sub)
            ws.write(3, 0, "", f_hdr)
            for j, c in enumerate(df.columns):
                ws.write(3, j + 1, str(c), f_hdr)
            for i, (lab, row) in enumerate(df.iterrows()):
                lab = str(lab)
                is_h, is_t = lab.startswith(HEAD), lab.startswith(TOTAL)
                text = lab.replace(HEAD, "").replace(TOTAL, "")
                ws.write(4 + i, 0, text, f_sec if is_h else (f_labtot if is_t else f_lab))
                for j, v in enumerate(row):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        if is_t:
                            ws.write_blank(4 + i, j + 1, None, f_tot)
                        continue
                    small = decimals or is_ratio_row(text)
                    ws.write_number(4 + i, j + 1, float(v), f_tot if is_t else (f_dec if small else f_num))
            ws.set_column(0, 0, 52)
            ws.set_column(1, len(df.columns), 15)
            ws.freeze_panes(4, 1)

        put("Income Statement", "Income statement & breakeven analysis (Annexure B)", income_table(res))
        put("Balance Sheet", "Balance sheet (Annexure C)", balance_table(res))
        for y in range(3):
            put(f"Cash Flow Y{y+1}", f"Monthly cash flow - year {y+1} (Annexure D{y+1})", cashflow_table(res, y))
        for y in range(3):
            put(f"Departments Y{y+1}", f"Monthly departmental accounts - year {y+1} (Annexure H{y+1})", dept_table(res, y))
        for y in range(3):
            put(f"Overheads Y{y+1}", f"Monthly overhead payments - year {y+1} (Annexure E{y+1})", overheads_table(res, y))
        for y in range(3):
            put(f"Assets Y{y+1}", f"Fixed asset register - year {y+1} (Annexure Q{y+1})", asset_table(res, y))
        L = res["loans"]
        for key, title in (("bp_new", "New Business Partners loans"), ("bp_exist", "Existing Business Partners loans"),
                           ("oth_exist", "Existing other loans"), ("oth_new", "New other loans")):
            if np.any(L[key]["balance"]) or np.any(L[key]["payout"]):
                put(title[:31], f"{title} - combined schedule", loan_table(res, L[key], 120), decimals=True)
        put("Ratios", "Key ratios", ratio_table(res))
        rt = res["returns"]
        rows = {"Total new Business Partners capital (R)": rt["total_capital"],
                "Administration & due diligence fee (R)": rt["admin_fee"],
                "Royalties over the deal (R, excl VAT)": rt["royalties_total"],
                "IRR - loan + fee (%)": None if rt["irr_loan_fee"] is None else rt["irr_loan_fee"] * 100,
                "IRR - total incl. royalties (%)": None if rt["irr_total"] is None else rt["irr_total"] * 100,
                "NPV of cash flows (R)": rt["npv"], "NPV of royalty fees (R)": rt["npv_royalty"],
                "Contribution above base rate (%)": None if rt["contribution_pct"] is None else rt["contribution_pct"] * 100,
                "Contribution (R)": rt["contribution_rand"]}
        put("Returns", "Risk and return (Annexure K / M extract)", pd.DataFrame({"Value": rows}), decimals=True)
    return buf.getvalue()
