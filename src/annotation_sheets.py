"""Spreadsheets for the human annotation (labels F, Q, quality type, evidence quotes).

Writes to annotation/:
  practice_round.xlsx      15 PROMISE items from development projects P02/P04 for all annotators (no answers shown)
  practice_key.csv         gold labels for those 15 items (for the study owner only; do not share before the meeting)
  main_round_template.xlsx empty sheet with the same columns, for the new external requirements

Practice items come from P02/P04 because those projects are development-only and never LLM test folds.
Item ids are neutral (PR-01 ...) so annotators cannot look up the gold labels.
"""
import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).parent))
import data as D

OUT = Path(__file__).resolve().parents[1] / "annotation"
ATTRS = sorted(set(D.ATTR_MAP.values()))
COLS = ["item_id", "requirement", "F", "Q", "quality_type", "F_quote", "Q_quote", "unsure", "note"]
WIDTH = {"item_id": 9, "requirement": 70, "F": 5, "Q": 5, "quality_type": 16, "F_quote": 28, "Q_quote": 28, "unsure": 8, "note": 30}
# 4 F only, 6 Q only (one per quality type except Maintainability), 5 F+Q
PRACTICE = ["P02-057", "P02-058", "P04-194", "P02-066",
            "P02-033", "P02-042", "P04-155", "P04-164", "P04-157", "P04-174",
            "P02-037", "P02-041", "P04-163", "P04-177", "P02-047"]
HELP = [
    "F = 1 if the requirement states something the system does (or lets a user do). Otherwise 0.",
    "Q = 1 if it states how well, how fast, how safely, how easily, or a limit (platform, law). Otherwise 0.",
    "quality_type: only when Q = 1. One of: " + ", ".join(ATTRS) + ".",
    "F_quote / Q_quote: copy the few key words that show the label, exactly as written. Empty when the label is 0.",
    "Never quote the whole requirement.",
    "unsure: yes if you are not confident; explain briefly in note.",
    "Work alone. Do not discuss items, and do not use ChatGPT or any other AI tool.",
]


def sheet(path, items):
    wb = Workbook(); ws = wb.active; ws.title = "labels"
    ws.append(COLS)
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="E7F1FB")
    for i, t in items:
        ws.append([i, t] + [None] * (len(COLS) - 2))
    for j, name in enumerate(COLS, 1):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = WIDTH[name]
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    last = max(ws.max_row, 300)
    col = {n: ws.cell(1, j).column_letter for j, n in enumerate(COLS, 1)}
    for name, options in (("F", "0,1"), ("Q", "0,1"), ("quality_type", ",".join(ATTRS)), ("unsure", "yes,no")):
        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"{col[name]}2:{col[name]}{last}")
    h = wb.create_sheet("help")
    h.column_dimensions["A"].width = 110
    for line in HELP:
        h.append([line])
    wb.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    P = D.promise().set_index("req_id")
    order = pd.Series(PRACTICE).sample(frac=1, random_state=7).tolist()  # mix the label types
    key = pd.DataFrame([{"item_id": f"PR-{n:02d}", "req_id": r, "text": P.loc[r, "text"], "F": int(P.loc[r, "F"]),
                         "Q": int(P.loc[r, "Q"]), "quality_type": P.loc[r, "attr"] if P.loc[r, "Q"] == 1 else ""}
                        for n, r in enumerate(order, 1)])
    assert set(P.loc[order, "project"]) <= {"P02", "P04"}
    assert not ((key.Q == 1) & (key.quality_type.isna() | (key.quality_type == ""))).any()
    sheet(OUT / "practice_round.xlsx", zip(key.item_id, key.text))
    key.to_csv(OUT / "practice_key.csv", index=False)
    sheet(OUT / "main_round_template.xlsx", [])
    print(key[["item_id", "req_id", "F", "Q", "quality_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
