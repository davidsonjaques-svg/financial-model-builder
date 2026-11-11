"""Import an existing BPL FinmodelSAP (Operating Entities) workbook into a deal dict.

Reads the *cached values* of the workbook (so the file must have been saved by Excel),
which means monthly figures that were typed over the template formulas are preserved.
"""
from __future__ import annotations
import datetime as dt
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter as L, column_index_from_string as CI

from engine import blank_deal, OVERHEAD_SLOTS, N_DEPTS, ASSET_CLASSES


def _num(v, default=0.0):
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _txt(v, default=""):
    if v is None:
        return default
    s = str(v).strip()
    return default if s in ("0", "None") else s


def _date(v, default=None):
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return default


def _row(ws, row, first_col, n=12):
    c0 = CI(first_col)
    return [_num(ws.cell(row=row, column=c0 + i).value) for i in range(n)]


def _row36(ws, row):
    """Months 1-36 on the DptCsh layout: B..M, R..AC, AG..AR."""
    return _row(ws, row, "B") + _row(ws, row, "R") + _row(ws, row, "AG")


def import_finmodel(path_or_file) -> dict:
    wb = load_workbook(path_or_file, data_only=True, keep_vba=False)
    need = {"DataInput", "DptCsh", "IStmt", "BSheet", "AstReg"}
    missing = need - set(wb.sheetnames)
    if missing:
        raise ValueError(f"Not a FinmodelSAP workbook - missing sheets: {sorted(missing)}")
    di, dc, is_, bs, ar = (wb[n] for n in ("DataInput", "DptCsh", "IStmt", "BSheet", "AstReg"))
    d = blank_deal()
    warnings = []

    # ---- general ----------------------------------------------------------
    g = d["general"]
    g["applicant"] = _txt(di["G6"].value, "Applicant")
    g["deal_type"] = _txt(di["G4"].value, "Loan Partner")
    g["fye"] = _date(di["G12"].value, g["fye"])
    if _num(di["G14"].value, 1) != 1:
        warnings.append("Workbook starts its projection mid-year (period "
                        f"{di['G14'].value}); this app projects from period 1 only.")
    a = d["assumptions"]
    a["tax_rate"] = _num(di["G58"].value, 0.27)
    a["npv_rate"] = _num(di["G59"].value, 0.105)
    a["target_irr"] = _num(di["G60"].value, 0.18)
    a["base_rate"] = _num(di["G61"].value, 0.105)
    a["overdraft_rate"] = _num(di["G62"].value, 0.115)
    a["prime_rate"] = _num(di["G63"].value, 0.105)
    a["overdraft_limit"] = _num(di["G64"].value)
    a["vat_ops"] = _num(di["G65"].value)
    a["vat_capex"] = _num(di["G66"].value)
    a["admin_fee_pct"] = _num(di["G41"].value, 0.015)
    a["sales_growth_after_y3"] = _num(di["G37"].value, 0.1)
    a["profit_growth_after_y3"] = _num(di["G39"].value, 0.08)
    d["royalty"]["tiers"] = [
        {"rate": _num(di[f"{c}34"].value), "until_month": int(_num(di[f"{c}35"].value))}
        for c in "GHI"]

    # ---- terms ------------------------------------------------------------
    t = d["terms"]
    t["debtors"] = [_num(dc[f"M{r}"].value) for r in range(3, 8)]
    t["creditors"] = [_num(dc[f"N{r}"].value) for r in range(3, 8)]

    # ---- loans ------------------------------------------------------------
    for i, c in enumerate("DEF"):
        ln = d["new_bp_loans"][i]
        ln["capital"] = _num(di[f"{c}72"].value)
        ln["term"] = int(_num(di[f"{c}73"].value))
        ln["rate"] = _num(di[f"{c}74"].value)
        ln["instalment"] = _num(di[f"{c}75"].value)
        ln["auto_instalment"] = False
        ln["tranches"] = [{"date": _date(di[f"{c}{r}"].value), "amount": _num(di[f"{c}{r+1}"].value)}
                          for r in (77, 79, 81, 83)]
        ln["first_instalment"] = _date(di[f"{c}87"].value)
        ln["balloon_period"] = int(_num(di[f"{c}89"].value))
        ln["balloon_amount"] = _num(di[f"{c}90"].value)
    for i, c in enumerate("DE"):
        ln = d["existing_bp_loans"][i]
        ln.update(balance=_num(di[f"{c}97"].value), instalment=_num(di[f"{c}98"].value),
                  rate=_num(di[f"{c}99"].value), redeem_period=int(_num(di[f"{c}105"].value)),
                  redeem_amount=_num(di[f"{c}106"].value))
    for i, c in enumerate("DEFG"):
        ln = d["existing_other_loans"][i]
        ln.update(balance=_num(di[f"{c}110"].value), instalment=_num(di[f"{c}111"].value),
                  rate=_num(di[f"{c}112"].value), redeem_period=int(_num(di[f"{c}118"].value)),
                  redeem_amount=_num(di[f"{c}119"].value))
    for i, c in enumerate("DE"):
        ln = d["new_other_loans"][i]
        ln.update(capital=_num(di[f"{c}123"].value), instalment=_num(di[f"{c}124"].value),
                  rate=_num(di[f"{c}125"].value), advance_date=_date(di[f"{c}128"].value),
                  first_instalment=_date(di[f"{c}130"].value))
    if any(_num(di[f"{c}134"].value) for c in "DEF") or _num(di["G72"].value) or _num(di["H72"].value):
        warnings.append("Workbook contains BEE shareholder loans, a BP shareholder loan or BP equity; "
                        "these instruments are not modelled in this app.")

    # ---- tax --------------------------------------------------------------
    periods = [int(_num(di[f"{c}157"].value, p)) for c, p in zip("EFG", (5, 6, 12))]
    d["tax"]["periods"] = periods
    for y, r in enumerate((159, 161, 163)):
        amts = [_num(di[f"{c}{r}"].value) for c in "EFG"]
        tot = sum(amts)
        d["tax"]["split"][y] = [x / tot for x in amts] if tot > 0 else [0.0, 0.0, 1.0]

    # ---- history ----------------------------------------------------------
    h = d["history"]
    h["years"] = [_date(is_[f"{c}8"].value) for c in "GIK"]
    for key, row in (("sales", 15), ("opening_stock", 18), ("purchases", 19), ("closing_stock", 20),
                     ("other_income", 26), ("depreciation", 50), ("interest_bp", 53),
                     ("interest_other", 54), ("tax", 57), ("dividends", 60)):
        h["is"][key] = [_num(is_[f"{c}{row}"].value) for c in "GIK"]
    names, methods, pcts, hist = [], [], [], []
    for r in range(30, 30 + OVERHEAD_SLOTS):
        names.append(_txt(is_[f"A{r}"].value))
        methods.append(int(_num(is_[f"B{r}"].value, 1)) or 1)
        pcts.append(_num(is_[f"C{r}"].value))
        hist.append([_num(is_[f"{c}{r}"].value) for c in "GIK"])
    d["overheads"]["names"] = names
    d["overheads"]["method"] = methods
    d["overheads"]["pct_of_sales"] = pcts
    h["is"]["overheads"] = hist
    for key, row in (("share_capital", 13), ("ndr", 14), ("retained_bf", 16), ("sh_loans", 17),
                     ("bp_loans", 23), ("other_loans", 25), ("other_loans_st", 26),
                     ("goodwill", 37), ("investments", 38), ("debtors", 43), ("bank", 44),
                     ("staff_loans", 45), ("tax_advance", 46), ("creditors", 50),
                     ("sundry_creditors", 51), ("overdraft", 52), ("receiver", 53), ("other_cl", 54)):
        h["bs"][key] = [_num(bs[f"{c}{row}"].value) for c in "CEG"]
    h["bs"]["bp_loans_st"] = [_num(bs[f"{c}24"].value) for c in "CEG"]
    h["bs"]["other_loans_st"] = [abs(x) for x in h["bs"]["other_loans_st"]]
    h["bs"]["bp_loans_st"] = [abs(x) for x in h["bs"]["bp_loans_st"]]
    h["bs"]["fa_land"] = [_num(bs[f"{c}32"].value) for c in "CEG"]
    h["bs"]["fa_equipment"] = [_num(bs[f"{c}33"].value) for c in "CEG"]
    h["bs"]["fa_vehicles"] = [_num(bs[f"{c}34"].value) for c in "CEG"]

    # ---- asset register ---------------------------------------------------
    for i, (top, _name) in enumerate(zip((5, 16, 25, 42), ASSET_CLASSES)):
        ac = d["assets"][i]
        ac["method"] = int(_num(ar[f"B{top+1}"].value, 2)) or 2
        ac["rate"] = _num(ar[f"B{top+2}"].value)
        cost = _num(ar[f"I{top+3}"].value) + _num(ar[f"I{top+4}"].value)
        bv = _num(ar[f"I{top+8}"].value)
        ac["opening_cost"] = cost
        ac["opening_book_value"] = bv
        ac["additions"] = (_row(ar, top + 4, "M") + _row(ar, top + 4, "AA") + _row(ar, top + 4, "AO"))

    # ---- monthly drivers --------------------------------------------------
    use_depts = _num(dc["D2"].value, 2) == 1
    dp = d["departments"]
    for i in range(N_DEPTS):
        dp["names"][i] = _txt(dc[f"A{9+i}"].value) if use_depts else ("Sales" if i == 0 else "")
        dp["sales"][i] = _row36(dc, 22 + i)
        dp["gp"][i] = _row36(dc, 30 + i)
    if not use_depts:
        # single department: back the GP% out of the total cost-of-sales row
        s, c = _row36(dc, 27), _row36(dc, 43)
        dp["sales"][0] = s
        dp["gp"][0] = [(1 - cc / ss) if ss else 0.0 for ss, cc in zip(s, c)]
        for i in range(1, N_DEPTS):
            dp["sales"][i] = [0.0] * 36
            dp["gp"][i] = [0.0] * 36
    m = d["monthly"]
    m["stock_increase"] = _row36(dc, 61)
    m["other_income"] = _row36(dc, 75)
    m["own_contribution"] = _row36(dc, 77)
    m["staff_loans"] = _row36(dc, 100)
    m["goodwill"] = _row36(dc, 101)
    m["dividends"] = _row36(dc, 112)
    d["overheads"]["monthly"] = [_row36(dc, 127 + i) for i in range(OVERHEAD_SLOTS)]
    d["overheads"]["escalation"] = [_num(dc["B124"].value), _num(dc["R124"].value), _num(dc["AG124"].value)]
    if any(_row36(dc, 111)):
        warnings.append("'Interest on Loan Account' cash-flow entries are not modelled and were ignored.")
    d["_import_warnings"] = warnings
    return d
