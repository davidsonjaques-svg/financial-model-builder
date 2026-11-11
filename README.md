# Financial Model Builder (operating entities)

A Streamlit version of the BPL **FinmodelSAP - Operating Only** workbook: capture a deal once,
get the 36-month cash flow, departmental accounts, income statement and breakeven, balance sheet,
asset register, loan schedules, key ratios and funder returns (IRR / NPV / contribution).

## Run

    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud exactly like the approval report generator: push this folder
to a (private) GitHub repo and point a new app at `app.py`.

## How it works

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (inputs on tabs 1-5, results, checks, save/export) |
| `engine.py` | All calculations - no Streamlit code, unit-testable |
| `statements.py` | Statement tables and the Excel pack |
| `importer.py` | Reads an existing FinmodelSAP `.xlsm` / `.xlsx` into a deal |
| `sample_deal_magand.json` | Sample deal (Magand Projects) - **client data, delete before sharing the repo** |
| `tests/` | `pytest tests/test_engine.py` checks the engine against the workbook's own results |

Workflow: import an existing workbook **or** start blank -> capture history -> (optionally) fill the
monthly grids from history -> refine month by month -> review Results and Checks -> download the
deal file (`.json`, reopen later) and the Excel pack.

## Scope

Modelled: projection from period 1 of the financial year; up to 5 departments; 18 overhead lines;
debtor / creditor terms; VAT on operations and capex; 4 asset classes (straight-line or reducing
balance); 3 new BP term loans (4 tranches, moratorium, balloon); 2 existing BP loans; 4 existing
and 2 new outside loans; overdraft interest; company tax with loss set-off; royalties; admin fee.

Not modelled (yet): mid-year projection start, BEE shareholder loans, BP shareholder loan / equity
and put-option exits (P/E sensitivities), manufacturing direct-cost split, the filling-station
section, the property schedule and the royalty/trend monitoring report.

Deliberate differences from the workbook: several tranches in the same month are added together;
tax after setting off a prior-year loss is floored at nil; tax paid mid-year is solved iteratively instead
of being typed in.
