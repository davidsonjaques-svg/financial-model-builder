import sys, json
sys.path.insert(0, '.')
from importer import import_finmodel
from engine import compute
from openpyxl import load_workbook
p = sys.argv[1]
d = import_finmodel(p)
print('warnings', d.pop('_import_warnings'))
json.dump(d, open('sample_deal_magand.json', 'w'), indent=1)
r = compute(d)
wb = load_workbook(p, data_only=True)
I, B, D, N = wb['IStmt'], wb['BSheet'], wb['DptCsh'], wb['Nic']
bad = 0
def chk(name, mine, xl, tol=1.0):
    global bad
    ok = abs(mine - xl) <= tol
    bad += (not ok)
    print(f"{'OK ' if ok else 'XX '}{name:38s} {mine:18,.2f} {xl:18,.2f}")
for y, c in enumerate('QSU'):
    for k, row in (('sales',15),('cos',16),('gross_profit',22),('overheads',28),('depreciation',50),('royalties',48),
                   ('npbit',52),('interest_bp',53),('interest_other',54),('npbt',56),('tax',57),('np_after_div',62),
                   ('capital_bp',66),('capital_other',67),('net_funds',69),('breakeven_cash',11),('breakeven_profit',12)):
        chk(f'IS Y{y+1} {k}', r['IS'][k][y], I[f'{c}{row}'].value)
for y, c in enumerate('MOQ'):
    for k, row in (('equity',12),('bp_loans',23),('other_loans',25),('fixed_assets',31),('debtors',43),('bank',44),
                   ('creditors',50),('overdraft',52),('receiver',53),('current_liabilities',48),('capital_employed',10),
                   ('employment_of_capital',29)):
        chk(f'BS Y{y+1} {k}', r['BS'][k][y], B[f'{c}{row}'].value)
    chk(f'BS Y{y+1} st_loans', r['BS']['st_loans'][y], B[f'{c}49'].value)
    chk(f'BS Y{y+1} difference', r['BS']['difference'][y], 0)
from openpyxl.utils import column_index_from_string as CI
for m in range(36):
    col = CI('B') + m if m < 12 else (CI('R') + m - 12 if m < 24 else CI('AG') + m - 24)
    chk(f'closing bank m{m+1}', r['closing_bank'][m], D.cell(row=118, column=col).value)
chk('hist NP y3', r['hist']['np_after_div'][2], I['K62'].value)
chk('IRR loan+fee %', r['returns']['irr_loan_fee']*100, wb['Loan']['AP2'].value*100, 0.01)
chk('IRR total %', r['returns']['irr_total']*100, N['D64'].value*100, 0.01)
chk('NPV', r['returns']['npv'], N['E64'].value)
chk('Contribution R', r['returns']['contribution_rand'], N['AD14'].value, 5)
print('MISMATCHES', bad)
