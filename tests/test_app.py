import json, sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("blank ok; metrics:", [(m.label, m.value) for m in at.metric[:6]])
# load sample
btn = [b for b in at.button if b.label == "Load sample"][0]
at = btn.click().run()
assert not at.exception, at.exception
print("sample ok; metrics:", [(m.label, m.value) for m in at.metric[:6]])
print("errors:", [e.value for e in at.error]); print("success:", len(at.success))
# change an input: overdraft rate
ni = [n for n in at.number_input if n.label.startswith("Overdraft interest rate")][0]
at = ni.set_value(20.0).run()
assert not at.exception, at.exception
print("after edit:", [(m.label, m.value) for m in at.metric[:6]])
# fill-from-history button
btn = [b for b in at.button if b.label == "Fill grids from history"][0]
at = btn.click().run()
assert not at.exception, at.exception
print("fill ok:", [(m.label, m.value) for m in at.metric[1:3]])
# rename a department -> refresh flag path
ti = [t for t in at.text_input if t.label == "Department 2"][0]
at = ti.set_value("Renamed dept").run()
assert not at.exception, at.exception
print("rename ok; version key:", at.session_state.ver)
