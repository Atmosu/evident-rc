"""Workbooks for the three annotators: the shared practice round plus the main round on the post-cutoff external set.

The 101 external requirements (data/external/post_cutoff_requirements.py) are shuffled, given neutral ids (EX-001 ...),
and split into three blocks; each block goes to one pair of annotators (A+B, B+C, A+C), so every item gets exactly
two independent labels and each annotator labels about 67 items. Row order is shuffled per annotator.

Writes to annotation/send/:   Annotator_A.xlsx, Annotator_B.xlsx, Annotator_C.xlsx   (send these)
          annotation/owner/:  external_items.csv (id, text, source, url, annotators)    (keep these)
"""
import random, sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/external")); sys.path.insert(0, str(Path(__file__).parent))
import post_cutoff_requirements as X
from annotation_sheets import ATTRS, COLS, WIDTH, HELP

SEND, OWNER = ROOT / "annotation/send", ROOT / "annotation/owner"
PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]


def fill(ws, items):
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
    col = {n: ws.cell(1, j).column_letter for j, n in enumerate(COLS, 1)}
    last = len(items) + 1
    for name, options in (("F", "0,1"), ("Q", "0,1"), ("quality_type", ",".join(ATTRS)), ("unsure", "yes,no")):
        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"{col[name]}2:{col[name]}{last}")


def main():
    SEND.mkdir(parents=True, exist_ok=True); OWNER.mkdir(parents=True, exist_ok=True)
    rows = [(src, t) for src, items in X.ITEMS.items() for t in items]
    random.Random(11).shuffle(rows)
    ext = pd.DataFrame([{"item_id": f"EX-{n:03d}", "requirement": t, "source": X.SOURCES[s][0], "url": X.SOURCES[s][1],
                         "source_code": s} for n, (s, t) in enumerate(rows, 1)])
    blocks = [ext.iloc[i::3] for i in range(3)]  # interleaved split keeps sources mixed in every block
    ext["annotators"] = ""
    for (a, b), blk in zip(PAIRS, blocks):
        ext.loc[blk.index, "annotators"] = a + "+" + b
    practice = pd.read_csv(ROOT / "annotation/practice_key.csv")
    for who in "ABC":
        mine = ext[ext.annotators.str.contains(who)].sample(frac=1, random_state=ord(who))
        wb = Workbook(); ws = wb.active; ws.title = "1_practice"
        fill(ws, list(zip(practice.item_id, practice.text)))
        fill(wb.create_sheet("2_main"), list(zip(mine.item_id, mine.requirement)))
        h = wb.create_sheet("help"); h.column_dimensions["A"].width = 110
        for line in ["Do tab 1_practice first (15 items), then tab 2_main.", *HELP,
                     "Save the file with your initials added to the name and send it back."]:
            h.append([line])
        wb.save(SEND / f"Annotator_{who}.xlsx")
        print(who, "practice", len(practice), "main", len(mine))
    ext.to_csv(OWNER / "external_items.csv", index=False)
    assert ext.annotators.str.len().eq(3).all() and ext.requirement.is_unique
    print(ext.annotators.value_counts().to_dict(), ext.source_code.value_counts().to_dict())


if __name__ == "__main__":
    main()
