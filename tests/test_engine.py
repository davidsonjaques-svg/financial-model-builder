"""Regression: the engine must reproduce the cached results of the Magand FinmodelSAP workbook."""
import json, os, sys
import pytest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))
from engine import compute, blank_deal  # noqa: E402

DEAL = json.load(open(os.path.join(HERE, "..", "sample_deal_magand.json")))
EXP = json.load(open(os.path.join(HERE, "expected_magand.json")))
RES = compute(DEAL)


@pytest.mark.parametrize("stmt", ["IS", "BS"])
def test_statements_match_workbook(stmt):
    for key, vals in EXP[stmt].items():
        for y in range(3):
            assert RES[stmt][key][y] == pytest.approx(vals[y], abs=1.0), (stmt, key, y)


def test_monthly_bank_balance_matches_workbook():
    for m in range(36):
        assert RES["closing_bank"][m] == pytest.approx(EXP["closing_bank"][m], abs=1.0), m


def test_returns_match_workbook():
    r = RES["returns"]
    assert r["irr_loan_fee"] == pytest.approx(EXP["irr_loan_fee"], abs=1e-5)
    assert r["irr_total"] == pytest.approx(EXP["irr_total"], abs=1e-5)
    assert r["npv"] == pytest.approx(EXP["npv"], abs=1.0)
    assert r["contribution_rand"] == pytest.approx(EXP["contribution_rand"], abs=5.0)


def test_balance_sheet_balances_under_stress():
    import copy
    d = copy.deepcopy(DEAL)
    d["assumptions"].update(vat_ops=0.15, vat_capex=0.15)
    d["assets"][1]["additions"][3] = 500000; d["assets"][3]["additions"][20] = 800000
    d["assets"][0].update(method=1, rate=0.02)
    d["terms"].update(debtors=[0.1, 0.5, 0.3, 0.1, 0.0], creditors=[0.3, 0.4, 0.2, 0.1, 0.0])
    d["tax"]["split"] = [[0.5, 0.0, 0.5]] * 3
    d["new_bp_loans"][1].update(capital=1500000, term=48, rate=0.14, auto_instalment=True,
                                first_instalment="2026-11-01", balloon_period=30, balloon_amount=200000)
    d["new_bp_loans"][1]["tranches"][0] = {"date": "2026-09-01", "amount": 1000000}
    d["new_bp_loans"][1]["tranches"][1] = {"date": "2026-12-01", "amount": 500000}
    d["new_other_loans"][0].update(capital=600000, instalment=20000, rate=0.12,
                                   advance_date="2026-05-01", first_instalment="2026-06-01")
    d["existing_other_loans"][0].update(redeem_period=8, redeem_amount=1000000)
    for k, m, v in (("own_contribution", 2, 250000), ("dividends", 18, 300000), ("stock_increase", 4, 400000),
                    ("goodwill", 6, 100000), ("staff_loans", 9, 50000)):
        d["monthly"][k][m] = v
    res = compute(d)
    assert max(abs(x) for x in res["BS"]["difference"]) < 0.01


def test_blank_deal_computes():
    res = compute(blank_deal())
    assert res["IS"]["sales"] == [0.0, 0.0, 0.0]
