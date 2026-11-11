"""Calculation engine for the Operating-Entity financial model (BPL FinmodelSAP logic).

Pure Python / numpy - no Streamlit imports - so it can be unit-tested and reused.

Conventions
-----------
* Months are numbered 1..36 from the first month of the financial year being projected
  (the engine keeps longer loan schedules internally for returns and short-term portions).
* A "deal" is a plain JSON-serialisable dict - see ``blank_deal``.
"""
from __future__ import annotations

import copy
import datetime as dt
from typing import Dict, List

import numpy as np

N_MONTHS = 36          # projection horizon shown in the statements
HORIZON = 168          # internal loan horizon (months)
N_DEPTS = 5
OVERHEAD_SLOTS = 18
ASSET_CLASSES = ["Land & Buildings", "Plant & Machinery", "Furniture & Fittings", "Motor Vehicles"]
DEFAULT_OVERHEADS = [
    "Advertising and Marketing", "Auditing and Accounting Fees", "Bad Debts",
    "Bank and Credit Card Charges", "Commission and Consulting Fees",
    "Electricity, Water and Property Tax", "Insurance and Security Costs",
    "Labour Related Expenses", "Leasing Charges and Hiring Costs", "Other Costs",
    "Owners' Remunerations", "Rent Paid", "Repairs, Maintenance and Cleaning",
    "Telephone, Postage, Printing and Stationery", "Transport, Delivery and Vehicle Costs",
    "", "", "",
]
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# --------------------------------------------------------------------------- dates
def _d(iso) -> dt.date:
    if isinstance(iso, dt.datetime):
        return iso.date()
    if isinstance(iso, dt.date):
        return iso
    return dt.date.fromisoformat(str(iso)[:10])


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    last = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
    return dt.date(y, m + 1, min(d.day, last))


def period_one(fye) -> dt.date:
    """First month of the first projected financial year."""
    return add_months(_d(fye), -11)


def month_index(date, fye) -> int:
    """Model month number (1 = first month of projected year 1) for a calendar date."""
    if not date:
        return 0
    d, p1 = _d(date), period_one(fye)
    return (d.year - p1.year) * 12 + (d.month - p1.month) + 1


def month_labels(fye, n=N_MONTHS) -> List[str]:
    p1 = period_one(fye)
    out = []
    for i in range(n):
        d = add_months(p1, i)
        out.append(f"{MONTH_ABBR[d.month - 1].title()}-{str(d.year)[2:]}")
    return out


def year_end_dates(fye) -> List[dt.date]:
    f = _d(fye)
    return [add_months(f, 12 * k) for k in range(3)]


# --------------------------------------------------------------------------- deal template
def blank_deal() -> dict:
    today = dt.date.today()
    fye = dt.date(today.year + (1 if today.month > 2 else 0), 2, 28)
    z36 = lambda: [0.0] * N_MONTHS  # noqa: E731
    z3 = lambda: [0.0, 0.0, 0.0]    # noqa: E731
    p1 = period_one(fye)
    return {
        "version": 1,
        "general": {"applicant": "", "deal_type": "Loan Partner", "fye": fye.isoformat(),
                    "prepared_by": "", "notes": ""},
        "assumptions": {"tax_rate": 0.27, "npv_rate": 0.105, "target_irr": 0.18, "base_rate": 0.105,
                        "overdraft_rate": 0.115, "prime_rate": 0.105, "overdraft_limit": 0.0,
                        "vat_ops": 0.0, "vat_capex": 0.0, "admin_fee_pct": 0.015,
                        "sales_growth_after_y3": 0.10, "profit_growth_after_y3": 0.08},
        "royalty": {"tiers": [{"rate": 0.0, "until_month": 0} for _ in range(3)]},
        "terms": {"debtors": [1.0, 0.0, 0.0, 0.0, 0.0], "creditors": [1.0, 0.0, 0.0, 0.0, 0.0]},
        "new_bp_loans": [{"capital": 0.0, "term": 60, "rate": 0.115, "instalment": 0.0,
                          "auto_instalment": True,
                          "tranches": [{"date": p1.isoformat(), "amount": 0.0} for _ in range(4)],
                          "first_instalment": add_months(p1, 1).isoformat(),
                          "balloon_period": 0, "balloon_amount": 0.0} for _ in range(3)],
        "existing_bp_loans": [{"balance": 0.0, "instalment": 0.0, "rate": 0.0,
                               "redeem_period": 0, "redeem_amount": 0.0} for _ in range(2)],
        "existing_other_loans": [{"balance": 0.0, "instalment": 0.0, "rate": 0.0,
                                  "redeem_period": 0, "redeem_amount": 0.0} for _ in range(4)],
        "new_other_loans": [{"capital": 0.0, "instalment": 0.0, "rate": 0.0,
                             "advance_date": p1.isoformat(),
                             "first_instalment": add_months(p1, 1).isoformat()} for _ in range(2)],
        "tax": {"periods": [5, 6, 12], "split": [[0.0, 0.0, 1.0] for _ in range(3)],
                "settle_opening_liability": True},
        "history": {
            "years": [add_months(fye, -36).isoformat(), add_months(fye, -24).isoformat(),
                      add_months(fye, -12).isoformat()],
            "is": {"sales": z3(), "opening_stock": z3(), "purchases": z3(), "closing_stock": z3(),
                   "other_income": z3(), "overheads": [z3() for _ in range(OVERHEAD_SLOTS)],
                   "depreciation": z3(), "interest_bp": z3(), "interest_other": z3(),
                   "tax": z3(), "dividends": z3()},
            "bs": {k: z3() for k in (
                "share_capital", "ndr", "retained_bf", "sh_loans", "bp_loans", "bp_loans_st",
                "other_loans", "other_loans_st", "fa_land", "fa_equipment", "fa_vehicles",
                "goodwill", "investments", "debtors", "bank", "staff_loans", "tax_advance",
                "creditors", "sundry_creditors", "overdraft", "receiver", "other_cl")},
        },
        "overheads": {"names": list(DEFAULT_OVERHEADS), "method": [1] * OVERHEAD_SLOTS,
                      "pct_of_sales": [0.0] * OVERHEAD_SLOTS,
                      "monthly": [z36() for _ in range(OVERHEAD_SLOTS)],
                      "escalation": [0.06, 0.06, 0.06]},
        "assets": [{"method": 2, "rate": r, "opening_cost": 0.0, "opening_book_value": 0.0,
                    "additions": z36()} for r in (0.0, 0.15, 0.15, 0.20)],
        "departments": {"names": ["Sales", "", "", "", ""],
                        "sales": [z36() for _ in range(N_DEPTS)],
                        "gp": [z36() for _ in range(N_DEPTS)]},
        "monthly": {k: z36() for k in ("stock_increase", "other_income", "own_contribution",
                                       "staff_loans", "goodwill", "dividends")},
    }


def normalise(deal: dict) -> dict:
    """Fill any keys missing from an older / partial deal file with template defaults."""
    base = blank_deal()

    def merge(b, d):
        if isinstance(b, dict) and isinstance(d, dict):
            out = {k: merge(b[k], d[k]) if k in d else copy.deepcopy(b[k]) for k in b}
            out.update({k: v for k, v in d.items() if k not in b})
            return out
        return copy.deepcopy(d)
    return merge(base, deal)


# --------------------------------------------------------------------------- finance helpers
def pmt(rate_pa: float, n: int, pv: float) -> float:
    if n <= 0 or pv == 0:
        return 0.0
    r = rate_pa / 12.0
    return pv / n if r == 0 else pv * r / (1 - (1 + r) ** -n)


def npv_excel(rate: float, flows) -> float:
    """Excel NPV(): the first cash flow is discounted one period."""
    f = np.asarray(flows, dtype=float)
    return float(np.sum(f / (1 + rate) ** np.arange(1, len(f) + 1)))


def irr(flows, lo=-0.99, hi=10.0):
    """Periodic IRR by bisection; returns None when there is no sign change."""
    f = np.asarray(flows, dtype=float)
    if not (np.any(f > 0) and np.any(f < 0)):
        return None
    t = np.arange(len(f))
    g = lambda r: float(np.sum(f / (1 + r) ** t))  # noqa: E731
    a, b = lo, hi
    ga, gb = g(a), g(b)
    if ga * gb > 0:
        return None
    for _ in range(200):
        m = (a + b) / 2
        gm = g(m)
        if abs(gm) < 1e-9:
            return m
        if ga * gm < 0:
            b, gb = m, gm
        else:
            a, ga = m, gm
    return (a + b) / 2


# --------------------------------------------------------------------------- loan schedules
def _blank_sched(h=HORIZON):
    return {k: np.zeros(h + 1) for k in ("payout", "interest", "capital", "payment", "balance", "redeem")}


def new_loan_schedule(rate, instalment, payouts: Dict[int, float], first_inst_idx: int,
                      balloon_period=0, balloon_amount=0.0, h=HORIZON):
    """Loan advanced during the projection: interest-only until the first instalment month."""
    s = _blank_sched(h)
    bal = 0.0
    for t in range(1, h + 1):
        po = payouts.get(t, 0.0)
        if balloon_period and t == balloon_period:
            po -= balloon_amount
        interest = bal * rate * 30 / 360
        if bal + interest < instalment:
            pay = bal + interest
        elif round(bal) <= 0:
            pay = 0.0
        elif t >= first_inst_idx:
            pay = instalment
        else:
            pay = interest
        bal = bal + po + interest - pay
        s["payout"][t], s["interest"][t], s["payment"][t] = po, interest, pay
        s["capital"][t], s["balance"][t] = pay - interest, bal
    return s


def existing_loan_schedule(balance, instalment, rate, redeem_period=0, redeem_amount=0.0, h=HORIZON):
    """Loan already on the balance sheet at the start of the projection."""
    s = _blank_sched(h)
    bal, prev_pay = float(balance), 0.0
    s["balance"][0] = bal
    for t in range(1, h + 1):
        if bal <= 1e-6:
            bal = 0.0
        interest = bal * rate * 30 / 360
        if bal + interest < prev_pay or instalment > bal:
            pay = bal + interest
        else:
            pay = instalment
        red = redeem_amount if (redeem_period and t == redeem_period) else 0.0
        red = min(red, max(bal - (pay - interest), 0.0))
        bal = bal - (pay - interest) - red
        s["interest"][t], s["payment"][t], s["capital"][t] = interest, pay, pay - interest
        s["redeem"][t], s["balance"][t] = red, bal
        prev_pay = pay
    return s


def _sum_scheds(scheds, h=HORIZON):
    out = _blank_sched(h)
    for s in scheds:
        for k in out:
            out[k] += s[k]
    return out


# --------------------------------------------------------------------------- asset register
def asset_schedule(method, rate, opening_cost, opening_bv, additions, n=N_MONTHS):
    """Monthly depreciation and book value.  method 1 = straight line, 2 = reducing balance."""
    add = np.asarray(additions, dtype=float)[:n]
    dep, bv = np.zeros(n), np.zeros(n)
    cost = float(opening_cost)
    prev_bv = float(opening_bv)
    for y in range(n // 12):
        base0 = cost if method == 1 else prev_bv          # depreciable base at start of year
        cum_add = 0.0
        for j in range(12):
            m = y * 12 + j
            d = (base0 + cum_add + add[m]) * rate / 12
            if d > prev_bv and add[m] < 1:
                d = prev_bv
            d = max(d, 0.0)
            cum_add += add[m]
            prev_bv = prev_bv + add[m] - d
            dep[m], bv[m] = d, prev_bv
        cost += cum_add
    return {"additions": add, "depreciation": dep, "book_value": bv}


# --------------------------------------------------------------------------- driver helper
def build_drivers_from_history(deal: dict, sales_growth, dept_split=None, dept_gp=None) -> dict:
    """Template defaults: project monthly sales and overheads off the last actual year.

    sales_growth : three annual growth rates (year 1 vs last actual, year 2 vs 1, year 3 vs 2)
    dept_split   : share of turnover per department (defaults to 100% in department 1)
    dept_gp      : GP % per department (defaults to last actual GP %)
    Overheads follow each line's method: 1 = escalate, 2 = % of sales, 3 = no increase.
    """
    d = copy.deepcopy(deal)
    h = d["history"]["is"]
    last_sales = h["sales"][2]
    cos = h["opening_stock"][2] + h["purchases"][2] - h["closing_stock"][2]
    hist_gp = (last_sales - cos) / last_sales if last_sales else 0.0
    split = list(dept_split or [1.0, 0, 0, 0, 0])
    gps = list(dept_gp or [hist_gp] * N_DEPTS)
    level, tot = last_sales / 12.0, np.zeros(N_MONTHS)
    for y in range(3):
        level *= (1 + sales_growth[y])
        for i in range(N_DEPTS):
            for j in range(12):
                d["departments"]["sales"][i][y * 12 + j] = level * split[i]
                d["departments"]["gp"][i][y * 12 + j] = gps[i] if split[i] else 0.0
        tot[y * 12:(y + 1) * 12] = level * sum(split)
    esc = d["overheads"]["escalation"]
    for i in range(OVERHEAD_SLOTS):
        base = h["overheads"][i][2] / 12.0
        meth = d["overheads"]["method"][i]
        lvl = base
        for y in range(3):
            if meth == 1:
                lvl *= (1 + esc[y])
            for j in range(12):
                m = y * 12 + j
                d["overheads"]["monthly"][i][m] = (
                    tot[m] * d["overheads"]["pct_of_sales"][i] if meth == 2 else lvl)
    return d


# --------------------------------------------------------------------------- main calculation
def compute(deal: dict) -> dict:
    d = normalise(deal)
    A, T = d["assumptions"], d["terms"]
    fye = d["general"]["fye"]
    n = N_MONTHS
    arr = lambda x: np.asarray(x, dtype=float)[:n]  # noqa: E731

    # ---- trading ------------------------------------------------------------------
    dept_sales = np.array([arr(r) for r in d["departments"]["sales"]])
    dept_gp = np.array([arr(r) for r in d["departments"]["gp"]])
    dept_cos = dept_sales * (1 - dept_gp)
    S, C = dept_sales.sum(axis=0), dept_cos.sum(axis=0)
    M = {k: arr(v) for k, v in d["monthly"].items()}
    K = M["stock_increase"]
    oh_lines = np.array([arr(r) for r in d["overheads"]["monthly"]])

    # ---- assets -------------------------------------------------------------------
    assets = [asset_schedule(a["method"], a["rate"], a["opening_cost"], a["opening_book_value"],
                             a["additions"]) for a in d["assets"]]
    dep = sum(a["depreciation"] for a in assets)
    ast_vat = sum(a["additions"] for a in assets) * A["vat_capex"]
    ast_vat_lag2 = np.concatenate([[0, 0], ast_vat[:-2]])

    # ---- loans --------------------------------------------------------------------
    bp_new_each, first_payout, morat_idx = [], None, []
    for ln in d["new_bp_loans"]:
        payouts: Dict[int, float] = {}
        for tr in ln["tranches"]:
            if tr.get("amount"):
                i = month_index(tr.get("date"), fye)
                if i >= 1:
                    payouts[i] = payouts.get(i, 0.0) + float(tr["amount"])
        inst = pmt(ln["rate"], int(ln["term"]), ln["capital"]) if ln.get("auto_instalment") else ln["instalment"]
        fi = max(month_index(ln.get("first_instalment"), fye), 1)
        bp_new_each.append(new_loan_schedule(ln["rate"], inst, payouts, fi,
                                             int(ln["balloon_period"] or 0), ln["balloon_amount"]))
        if payouts:
            first_payout = min(payouts) if first_payout is None else min(first_payout, min(payouts))
            morat_idx.append(fi)
    bp_exist_each = [existing_loan_schedule(x["balance"], x["instalment"], x["rate"],
                                            int(x["redeem_period"] or 0), x["redeem_amount"])
                     for x in d["existing_bp_loans"]]
    oth_exist_each = [existing_loan_schedule(x["balance"], x["instalment"], x["rate"],
                                             int(x["redeem_period"] or 0), x["redeem_amount"])
                      for x in d["existing_other_loans"]]
    oth_new_each = []
    for x in d["new_other_loans"]:
        ai = month_index(x.get("advance_date"), fye)
        po = {ai: float(x["capital"])} if (x["capital"] and ai >= 1) else {}
        oth_new_each.append(new_loan_schedule(x["rate"], x["instalment"], po,
                                              max(month_index(x.get("first_instalment"), fye), 1)))
    bp_new, bp_exist = _sum_scheds(bp_new_each), _sum_scheds(bp_exist_each)
    oth_exist, oth_new = _sum_scheds(oth_exist_each), _sum_scheds(oth_new_each)
    total_capital = sum(float(x["capital"]) for x in d["new_bp_loans"])

    # ---- admin fee & royalties ----------------------------------------------------
    admin_fee = np.zeros(HORIZON + 1)
    if first_payout:
        admin_fee[first_payout] = total_capital * A["admin_fee_pct"]
    S_ext = np.zeros(HORIZON + 1)
    S_ext[1:n + 1] = S
    for t in range(n + 1, HORIZON + 1):                 # turnover beyond year 3
        S_ext[t] = S_ext[t - 12] * (1 + A["sales_growth_after_y3"])
    tiers = d["royalty"]["tiers"]

    def roy_rate(t):
        for tier in tiers:
            if t <= tier["until_month"]:
                return tier["rate"]
        return 0.0
    roy_start = (max(morat_idx) - 1) if morat_idx else None
    roy_sales, roy_incl = np.zeros(HORIZON + 1), np.zeros(HORIZON + 1)
    if roy_start is not None:
        for t in range(1, HORIZON + 1):
            roy_sales[t] = S_ext[t] if t >= roy_start else 0.0
        for t in range(2, HORIZON + 1):
            roy_incl[t] = roy_sales[t - 1] * roy_rate(t - 1) * (1 + A["vat_capex"])
    roy_vat = roy_incl * A["vat_capex"] / (1 + A["vat_capex"])
    roy_excl = roy_incl - roy_vat

    overheads_cash = oh_lines.sum(axis=0) + admin_fee[1:n + 1] + roy_excl[1:n + 1]

    # ---- opening balance sheet ----------------------------------------------------
    hb = {k: v[2] for k, v in d["history"]["bs"].items()}
    hist = historic_income(d)
    open_bank = hb["bank"] - hb["overdraft"]

    # ---- cash-flow building blocks that do not depend on the bank balance ----------
    dt_, ct_ = T["debtors"], T["creditors"]

    def spread(terms):
        cash, t30, t60, t90, t120 = terms
        if cash == 1:
            return [1.0, 0.0, 0.0, 0.0]
        den = t30 + 2 * t60 + 3 * t90 + 4 * t120
        if den == 0:
            return [1.0, 0.0, 0.0, 0.0]
        return [(t30 + t60 + t90 + t120) / den, (t60 + t90 + t120) / den, (t90 + t120) / den, t120 / den]

    def lag(x, k):
        return np.concatenate([np.zeros(k), x[:-k]]) if k else x.copy()

    r = {}
    r["Existing Debtors"] = np.zeros(n)
    r["Existing Debtors"][:4] = hb["debtors"] * np.array(spread(dt_))
    vat_out = ast_vat + S * A["vat_ops"] + roy_vat[1:n + 1]                 # "Value Added Tax" payment
    vat_in = (K + C) * A["vat_ops"] + ast_vat_lag2 + roy_vat[1:n + 1]       # "Value Added Tax" receipt
    r["Cash Sales"] = S * dt_[0] + vat_out - ast_vat
    for k, lab in enumerate(("30", "60", "90", "120"), start=1):
        r[f"Debtors - {lab} Days"] = lag(S, k) * dt_[k]
    r["Value Added Tax"] = vat_in
    r["Other Income"] = M["other_income"]
    r["Other Loans"] = oth_new["payout"][1:n + 1]
    r["Own Contribution"] = M["own_contribution"]
    r["Term Loans: Business Partners"] = bp_new["payout"][1:n + 1]
    receipts = sum(r.values())

    p = {}
    p["Existing Creditors"] = np.zeros(n)
    p["Existing Creditors"][:4] = hb["creditors"] * np.array(spread(ct_))
    p["Cash Purchases"] = (C + K) * ct_[0] + vat_in - ast_vat_lag2
    for k, lab in enumerate(("30", "60", "90", "120"), start=1):
        p[f"Creditors - {lab} Days"] = lag(C + K, k) * ct_[k]
    p["Overhead Payments"] = overheads_cash
    p["Interest on Overdraft"] = np.zeros(n)
    p["Land & Buildings"] = assets[0]["additions"]
    p["Furniture & Equipment"] = assets[1]["additions"] + assets[2]["additions"]
    p["Motor Vehicles"] = assets[3]["additions"]
    p["Value Added Tax "] = vat_out
    p["Staff Loans & Deposits"] = M["staff_loans"]
    p["Goodwill"] = M["goodwill"]
    p["Other Loans - Interest"] = (oth_exist["interest"] + oth_new["interest"])[1:n + 1]
    p["Other Loans - Capital"] = (oth_exist["capital"] + oth_exist["redeem"] + oth_new["capital"])[1:n + 1]
    p["Business Partners Exist Loans - Interest"] = bp_exist["interest"][1:n + 1]
    p["Business Partners Exist Loans - Capital"] = (bp_exist["capital"] + bp_exist["redeem"])[1:n + 1]
    p["Business Partners New Loans - Interest"] = bp_new["interest"][1:n + 1]
    p["Business Partners New Loans - Capital"] = bp_new["capital"][1:n + 1]
    p["Taxation Paid"] = np.zeros(n)
    p["Dividends"] = M["dividends"]
    fixed_payments = sum(v for k, v in p.items() if k not in ("Interest on Overdraft", "Taxation Paid"))

    yr = lambda x, y: float(np.sum(x[y * 12:(y + 1) * 12]))  # noqa: E731
    int_bp_m = p["Business Partners Exist Loans - Interest"] + p["Business Partners New Loans - Interest"]
    periods, split = d["tax"]["periods"], d["tax"]["split"]
    open_tax_liab = hb["receiver"] - hb["tax_advance"]

    # ---- bank loop, iterated because tax paid <-> overdraft interest are circular --
    tax_paid = np.zeros(n)
    for _ in range(40):
        od_int, opening, closing = np.zeros(n), np.zeros(n), np.zeros(n)
        bal = open_bank
        for m in range(n):
            opening[m] = bal
            if bal < 0:
                od_int[m] = -bal * A["overdraft_rate"] / 12
                if m >= 12:
                    od_int[m] = round(od_int[m])
            bal += receipts[m] - fixed_payments[m] - od_int[m] - tax_paid[m]
            closing[m] = bal
        int_other_m = p["Other Loans - Interest"] + od_int
        npbt_m = S - C + M["other_income"] - overheads_cash - dep - int_bp_m - int_other_m
        npbt = [yr(npbt_m, y) for y in range(3)]
        tax = [0.0, 0.0, 0.0]
        tax[0] = max(npbt[0] * A["tax_rate"], 0.0)
        for y in (1, 2):
            if npbt[y] > 0:                      # prior-year loss is set off against this year
                base = npbt[y] if npbt[y - 1] > 0 else npbt[y] + npbt[y - 1]
                tax[y] = max(base * A["tax_rate"], 0.0)
        new_paid = np.zeros(n)
        for y in range(3):
            due = tax[y] + (open_tax_liab if (y == 0 and d["tax"].get("settle_opening_liability", True)) else 0.0)
            for k, per in enumerate(periods):
                if 1 <= per <= 12:
                    new_paid[y * 12 + per - 1] += due * split[y][k]
        # stop once stable; the pass just completed used `tax_paid`, so keep it (any cent-level
        # residual from the rounded overdraft interest simply stays in Receiver of Revenue)
        if _ == 39 or np.allclose(new_paid, tax_paid, rtol=0, atol=0.005):
            break
        tax_paid = new_paid
    p["Interest on Overdraft"], p["Taxation Paid"] = od_int, tax_paid
    payments = sum(p.values())

    # ---- income statement ---------------------------------------------------------
    cap_bp_m = p["Business Partners Exist Loans - Capital"] + p["Business Partners New Loans - Capital"]
    cap_oth_m = p["Other Loans - Capital"]
    stock_open = hist["closing_stock"][2]
    IS = {k: [0.0] * 3 for k in (
        "sales", "cos", "opening_stock", "purchases", "closing_stock", "gross_profit", "gp_pct",
        "other_income", "overheads", "depreciation", "royalties", "admin_fee", "npbit", "interest_bp",
        "interest_other", "npbt", "tax", "np_after_tax", "dividends", "np_after_div",
        "funds_generated", "capital_bp", "capital_other", "net_funds", "breakeven_cash",
        "breakeven_profit", "avg_turnover")}
    IS["overhead_lines"] = [[yr(oh_lines[i], y) for y in range(3)] for i in range(OVERHEAD_SLOTS)]
    for y in range(3):
        I = lambda k, v: IS[k].__setitem__(y, v)  # noqa: E731,E741
        I("sales", yr(S, y)); I("cos", yr(C, y))
        I("opening_stock", stock_open); I("purchases", yr(C, y) + yr(K, y))
        stock_open = stock_open + yr(K, y)
        I("closing_stock", stock_open)
        gp = yr(S, y) - yr(C, y)
        I("gross_profit", gp); I("gp_pct", gp / yr(S, y) if yr(S, y) else 0.0)
        I("other_income", yr(M["other_income"], y))
        I("royalties", yr(roy_excl[1:n + 1], y)); I("admin_fee", yr(admin_fee[1:n + 1], y))
        I("depreciation", yr(dep, y))
        oh = yr(overheads_cash, y) + yr(dep, y)
        I("overheads", oh)
        I("npbit", gp + IS["other_income"][y] - oh)
        I("interest_bp", yr(int_bp_m, y)); I("interest_other", yr(int_other_m, y))
        I("npbt", IS["npbit"][y] - IS["interest_bp"][y] - IS["interest_other"][y])
        I("tax", tax[y]); I("np_after_tax", IS["npbt"][y] - tax[y])
        I("dividends", yr(M["dividends"], y)); I("np_after_div", IS["np_after_tax"][y] - IS["dividends"][y])
        I("funds_generated", IS["np_after_div"][y] + IS["depreciation"][y])
        I("capital_bp", yr(cap_bp_m, y)); I("capital_other", yr(cap_oth_m, y))
        I("net_funds", IS["funds_generated"][y] - IS["capital_bp"][y] - IS["capital_other"][y])
        I("avg_turnover", yr(S, y) / 12)
        g = IS["gp_pct"][y]
        intr, cap = IS["interest_bp"][y] + IS["interest_other"][y], IS["capital_bp"][y] + IS["capital_other"][y]
        I("breakeven_profit", (oh + intr - IS["other_income"][y]) / g / 12 if g else 0.0)
        I("breakeven_cash", ((oh - IS["depreciation"][y] + intr + cap - IS["other_income"][y])
                             + cap * A["tax_rate"]) / g / 12 if g else 0.0)

    # ---- balance sheet ------------------------------------------------------------
    cash_d, cash_c = dt_[0], ct_[0]
    coll = sum(r[k] for k in r if k.startswith("Debtors -")) + r["Existing Debtors"]
    paid_c = sum(p[k] for k in p if k.startswith("Creditors -")) + p["Existing Creditors"]
    debt_m = hb["debtors"] + np.cumsum(S * (1 - cash_d) - coll) + (ast_vat + lag(ast_vat, 1))
    cred_m = hb["creditors"] + np.cumsum((C + K) * (1 - cash_c) - paid_c)
    bp_bal = bp_new["balance"] + bp_exist["balance"]
    oth_bal = oth_exist["balance"] + np.maximum(oth_new["balance"], 0)
    bp_cap_all = bp_new["capital"] + bp_exist["capital"] + bp_exist["redeem"]
    oth_pay_all = oth_exist["payment"] + oth_new["payment"]

    BS = {k: [0.0] * 3 for k in (
        "share_capital", "ndr", "net_income", "retained_bf", "sh_loans", "equity", "bp_loans",
        "bp_loans_st", "other_loans", "other_loans_st", "lt_loans", "capital_employed", "fa_land",
        "fa_equipment", "fa_vehicles", "fixed_assets", "goodwill", "investments", "lt_assets",
        "stock", "debtors", "bank", "staff_loans", "tax_advance", "current_assets", "st_loans",
        "creditors", "sundry_creditors", "overdraft", "receiver", "other_cl",
        "current_liabilities", "net_current_assets", "employment_of_capital", "difference")}
    retained = hb["retained_bf"] + hist["np_after_div"][2]
    receiver = open_tax_liab
    for y in range(3):
        e = (y + 1) * 12                       # month number at year end
        B = lambda k, v: BS[k].__setitem__(y, v)  # noqa: E731
        B("share_capital", hb["share_capital"]); B("ndr", hb["ndr"])
        B("net_income", IS["np_after_div"][y]); B("retained_bf", retained)
        retained += IS["np_after_div"][y]
        B("sh_loans", hb["sh_loans"] + float(np.sum(M["own_contribution"][:e])))
        B("equity", sum(BS[k][y] for k in ("share_capital", "ndr", "net_income", "retained_bf", "sh_loans")))
        B("bp_loans", bp_bal[e]); B("bp_loans_st", min(float(np.sum(bp_cap_all[e + 1:e + 13])), bp_bal[e]))
        B("other_loans", oth_bal[e])
        # template convention: next 12 instalments (year 3: month-37 instalment x 12)
        nxt = float(oth_pay_all[e + 1]) * 12 if y == 2 else float(np.sum(oth_pay_all[e + 1:e + 13]))
        B("other_loans_st", min(nxt, oth_bal[e]))
        B("lt_loans", BS["bp_loans"][y] - BS["bp_loans_st"][y] + BS["other_loans"][y] - BS["other_loans_st"][y])
        B("capital_employed", BS["equity"][y] + BS["lt_loans"][y])
        B("fa_land", assets[0]["book_value"][e - 1])
        B("fa_equipment", assets[1]["book_value"][e - 1] + assets[2]["book_value"][e - 1])
        B("fa_vehicles", assets[3]["book_value"][e - 1])
        B("fixed_assets", BS["fa_land"][y] + BS["fa_equipment"][y] + BS["fa_vehicles"][y])
        B("goodwill", hb["goodwill"] + float(np.sum(M["goodwill"][:e]))); B("investments", hb["investments"])
        B("lt_assets", BS["goodwill"][y] + BS["investments"][y])
        B("stock", IS["closing_stock"][y]); B("debtors", debt_m[e - 1])
        B("bank", closing[e - 1] if closing[e - 1] >= 1 else 0.0)
        B("overdraft", -closing[e - 1] if closing[e - 1] < 1 else 0.0)
        B("staff_loans", hb["staff_loans"] + float(np.sum(M["staff_loans"][:e]))); B("tax_advance", 0.0)
        B("current_assets", sum(BS[k][y] for k in ("stock", "debtors", "bank", "staff_loans", "tax_advance")))
        B("st_loans", BS["bp_loans_st"][y] + BS["other_loans_st"][y])
        B("creditors", cred_m[e - 1]); B("sundry_creditors", hb["sundry_creditors"])
        receiver = receiver + tax[y] - yr(tax_paid, y)
        B("receiver", receiver); B("other_cl", hb["other_cl"])
        B("current_liabilities", sum(BS[k][y] for k in ("st_loans", "creditors", "sundry_creditors",
                                                        "overdraft", "receiver", "other_cl")))
        B("net_current_assets", BS["current_assets"][y] - BS["current_liabilities"][y])
        B("employment_of_capital", BS["fixed_assets"][y] + BS["lt_assets"][y] + BS["net_current_assets"][y])
        B("difference", BS["employment_of_capital"][y] - BS["capital_employed"][y])

    # ---- returns to the funder ----------------------------------------------------
    returns = {"irr_loan_fee": None, "irr_total": None, "npv": 0.0, "npv_royalty": 0.0,
               "contribution_pct": None, "contribution_rand": None, "flows": [], "total_capital": total_capital,
               "admin_fee": float(admin_fee.sum()), "royalties_total": float(roy_excl.sum())}
    if first_payout:
        t0 = first_payout
        base = -bp_new["payout"][t0:] + admin_fee[t0:] + bp_new["payment"][t0:]
        full = base + roy_excl[t0:]
        last = int(np.max(np.nonzero(full)[0])) + 1 if np.any(full) else 1
        base, full = base[:last], full[:last]
        i1, i2 = irr(base), irr(full)
        returns.update(
            irr_loan_fee=None if i1 is None else i1 * 12, irr_total=None if i2 is None else i2 * 12,
            npv=npv_excel(A["npv_rate"] / 12, full), npv_royalty=npv_excel(A["npv_rate"] / 12, roy_excl[t0:t0 + last]),
            flows=full.tolist(), first_payout=t0)
        if i2 is not None:
            returns["contribution_pct"] = i2 * 12 - A["base_rate"]
            returns["contribution_rand"] = returns["contribution_pct"] * total_capital

    # ---- ratios -------------------------------------------------------------------
    ratios = {k: [None] * 3 for k in ("gp_pct", "np_pct", "current_ratio", "acid_test", "gearing",
                                      "interest_cover", "dscr", "debtor_days", "creditor_days",
                                      "peak_overdraft", "min_headroom")}
    div = lambda a, b: (a / b) if b else None  # noqa: E731
    for y in range(3):
        R = lambda k, v: ratios[k].__setitem__(y, v)  # noqa: E731
        intr = IS["interest_bp"][y] + IS["interest_other"][y]
        cap = IS["capital_bp"][y] + IS["capital_other"][y]
        R("gp_pct", IS["gp_pct"][y]); R("np_pct", div(IS["np_after_tax"][y], IS["sales"][y]))
        R("current_ratio", div(BS["current_assets"][y], BS["current_liabilities"][y]))
        R("acid_test", div(BS["current_assets"][y] - BS["stock"][y], BS["current_liabilities"][y]))
        debt = BS["bp_loans"][y] + BS["other_loans"][y] + BS["overdraft"][y]
        R("gearing", div(debt, BS["equity"][y]))
        R("interest_cover", div(IS["npbit"][y], intr))
        R("dscr", div(IS["npbit"][y] + IS["depreciation"][y] - IS["tax"][y], intr + cap))
        R("debtor_days", div(BS["debtors"][y] * 365, IS["sales"][y] * (1 - cash_d)) if cash_d < 1 else 0.0)
        R("creditor_days", div(BS["creditors"][y] * 365, IS["purchases"][y] * (1 - cash_c)) if cash_c < 1 else 0.0)
        lo = float(np.min(closing[y * 12:(y + 1) * 12]))
        R("peak_overdraft", max(-lo, 0.0)); R("min_headroom", A["overdraft_limit"] + lo)

    return {
        "deal": d, "labels": month_labels(fye), "year_ends": year_end_dates(fye),
        "dept": {"sales": dept_sales, "gp": dept_gp, "cos": dept_cos},
        "sales": S, "cos": C, "stock_increase": K, "receipts": r, "payments": p,
        "total_receipts": receipts, "total_payments": payments, "opening_bank": opening,
        "closing_bank": closing, "overhead_lines": oh_lines, "overheads_cash": overheads_cash,
        "royalties": roy_excl[1:n + 1], "admin_fee": admin_fee[1:n + 1], "depreciation": dep,
        "npbt_monthly": npbt_m, "assets": assets, "hist": hist, "IS": IS, "BS": BS,
        "loans": {"bp_new": bp_new, "bp_exist": bp_exist, "oth_exist": oth_exist, "oth_new": oth_new,
                  "bp_new_each": bp_new_each, "bp_exist_each": bp_exist_each,
                  "oth_exist_each": oth_exist_each, "oth_new_each": oth_new_each},
        "returns": returns, "ratios": ratios, "overdraft_limit": A["overdraft_limit"],
    }


def historic_income(deal: dict) -> dict:
    """Derived lines of the three actual income statements."""
    h = deal["history"]["is"]
    out = {k: list(h[k]) for k in ("sales", "opening_stock", "purchases", "closing_stock", "other_income",
                                   "depreciation", "interest_bp", "interest_other", "tax", "dividends")}
    out["overhead_lines"] = [list(x) for x in h["overheads"]]
    keys = ("cos", "gross_profit", "gp_pct", "overheads", "npbit", "npbt", "np_after_tax", "np_after_div",
            "funds_generated", "avg_turnover")
    for k in keys:
        out[k] = [0.0] * 3
    for y in range(3):
        cos = h["opening_stock"][y] + h["purchases"][y] - h["closing_stock"][y]
        gp = h["sales"][y] - cos
        oh = sum(x[y] for x in h["overheads"]) + h["depreciation"][y]
        npbit = gp + h["other_income"][y] - oh
        npbt = npbit - h["interest_bp"][y] - h["interest_other"][y]
        nat = npbt - h["tax"][y]
        out["cos"][y], out["gross_profit"][y] = cos, gp
        out["gp_pct"][y] = gp / h["sales"][y] if h["sales"][y] else 0.0
        out["overheads"][y], out["npbit"][y], out["npbt"][y] = oh, npbit, npbt
        out["np_after_tax"][y], out["np_after_div"][y] = nat, nat - h["dividends"][y]
        out["funds_generated"][y] = nat - h["dividends"][y] + h["depreciation"][y]
        out["avg_turnover"][y] = h["sales"][y] / 12
    return out
