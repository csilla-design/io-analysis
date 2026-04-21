"""Compute output and labor-income multipliers for Thailand's 5 industries of the future.

Input: IO2021 180 sectors.xlsx
  - DataIO2021 sheet: 180-sector input-output table in long format with columns
    ROW, ROW description, COLUMN, COLUMN description, PURCHASER, WHOLESALE, RETAIL,
    TRANSPORT, IMPORT. Purchaser price = producer price + trade margins + transport;
    producer price = domestic + imported.
  - sector mapping sheet: maps each of 180 IO sectors to one of the 5 industries of
    the future (Agribusiness, Creative industries, Advanced manufacturing, Tourism,
    Digital services).

Method (standard Leontief open model, Type I):
  - Domestic intermediate flows at producer prices:
        Z[i,j] = PURCHASER - WHOLESALE - RETAIL - TRANSPORT - IMPORT
  - Total output x[j] from row 210 "Control total" for each industry column.
  - Technical-coefficient matrix A[i,j] = Z[i,j] / x[j].
  - Leontief inverse L = (I - A)^{-1}.
  - Output multiplier for sector j: m_j = sum_i L[i,j].
  - Labor coefficient h_i = wages_salaries_i / x_i (row 201 / total output).
  - Labor-income multiplier for sector j: mi_j = sum_i h_i * L[i,j].
    NB: A true employment multiplier requires employment (headcount) by sector,
    which is NOT present in the IO file. We report wages as the available labor
    proxy and flag this limitation.

Aggregation to the 5 industries of the future:
  - Each industry multiplier is reported as the output-weighted average of its
    member sector multipliers, using sector total outputs as weights.
"""

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

SRC = "IO2021 180 sectors.xlsx"
OUT = "industries_of_the_future_multipliers.xlsx"

wb = openpyxl.load_workbook(SRC, data_only=True)
data = wb["DataIO2021"]
mapping = wb["sector mapping"]

# --- Sector universe: codes 001..180 ---
sector_codes = [f"{i:03d}" for i in range(1, 181)]
code_to_idx = {c: i for i, c in enumerate(sector_codes)}
N = len(sector_codes)

# --- Read long-format IO data into Z, total output x, wages w ---
Z = np.zeros((N, N), dtype=float)       # domestic intermediate flows
x = np.zeros(N, dtype=float)            # total output (row 210)
w = np.zeros(N, dtype=float)            # wages and salaries (row 201)
sector_name = {c: None for c in sector_codes}

for row in data.iter_rows(min_row=3, values_only=True):
    rc, rd, cc, cd, purch, whole, retail, trans, imp = row
    if rc is None:
        break
    purch = purch or 0
    whole = whole or 0
    retail = retail or 0
    trans = trans or 0
    imp = imp or 0

    if cc in code_to_idx:
        j = code_to_idx[cc]
        if sector_name[cc] is None:
            sector_name[cc] = cd
        if rc in code_to_idx:
            i = code_to_idx[rc]
            # domestic flow at producer price
            Z[i, j] = purch - whole - retail - trans - imp
        elif rc == "201":
            w[j] = purch
        elif rc == "210":
            x[j] = purch

# --- Build A and Leontief inverse ---
# Avoid division by zero for sectors with zero output
A = np.zeros_like(Z)
nonzero = x > 0
A[:, nonzero] = Z[:, nonzero] / x[nonzero]

I = np.eye(N)
L = np.linalg.inv(I - A)

# --- Output multiplier: column sums of L ---
output_mult = L.sum(axis=0)

# --- Labor-income multiplier: h'·L where h_i = w_i / x_i ---
h = np.zeros(N)
h[nonzero] = w[nonzero] / x[nonzero]
# Direct+indirect labor income per unit of sector j's output:
labor_income_per_output = h @ L
# Type I labor-income multiplier = (direct+indirect) / direct
with np.errstate(divide="ignore", invalid="ignore"):
    labor_income_mult_typeI = np.where(h > 0, labor_income_per_output / h, np.nan)

# --- Industry mapping ---
industry_of = {}
for row in mapping.iter_rows(min_row=2, values_only=True):
    code, name, industry = row[0], row[1], row[2]
    if code and industry:
        industry_of[code] = industry

industries_order = [
    "Agribusiness",
    "Creative industries",
    "Advanced manufacturing",
    "Tourism",
    "Digital services",
]

# --- Aggregate to 5 industries: output-weighted averages across member sectors ---
def weighted_avg(values, weights):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (weights > 0)
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))

industry_rows = []
for ind in industries_order:
    members = [c for c, v in industry_of.items() if v == ind]
    idxs = [code_to_idx[c] for c in members]
    x_members = x[idxs]
    om = weighted_avg(output_mult[idxs], x_members)
    lim = weighted_avg(labor_income_mult_typeI[idxs], x_members)
    # Also report simple (unweighted) average
    om_simple = float(np.nanmean(output_mult[idxs]))
    lim_simple = float(np.nanmean(labor_income_mult_typeI[idxs]))
    industry_rows.append({
        "industry": ind,
        "n_sectors": len(members),
        "total_output": float(x_members.sum()),
        "total_wages": float(w[idxs].sum()),
        "output_mult_weighted": om,
        "output_mult_simple_avg": om_simple,
        "labor_income_mult_weighted": lim,
        "labor_income_mult_simple_avg": lim_simple,
    })

# --- Write output workbook ---
out = openpyxl.Workbook()
ws = out.active
ws.title = "Industry summary"

header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="305496")
bold = Font(bold=True)

ws["A1"] = "Output and labor-income multipliers, Thailand IO 2021 (180 sectors)"
ws["A1"].font = Font(bold=True, size=13)
ws.merge_cells("A1:H1")

ws["A3"] = "Aggregated to the 5 industries of the future (weighted by sector total output)."
ws.merge_cells("A3:H3")

headers = [
    "Industry of the future",
    "# sectors",
    "Total output (million baht)",
    "Total wages (million baht)",
    "Output multiplier (weighted avg)",
    "Output multiplier (simple avg)",
    "Labor-income multiplier Type I (weighted avg)",
    "Labor-income multiplier Type I (simple avg)",
]
for j, h_ in enumerate(headers, start=1):
    c = ws.cell(row=5, column=j, value=h_)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")

for r, row in enumerate(industry_rows, start=6):
    ws.cell(row=r, column=1, value=row["industry"]).font = bold
    ws.cell(row=r, column=2, value=row["n_sectors"])
    ws.cell(row=r, column=3, value=row["total_output"]).number_format = "#,##0"
    ws.cell(row=r, column=4, value=row["total_wages"]).number_format = "#,##0"
    ws.cell(row=r, column=5, value=row["output_mult_weighted"]).number_format = "0.0000"
    ws.cell(row=r, column=6, value=row["output_mult_simple_avg"]).number_format = "0.0000"
    ws.cell(row=r, column=7, value=row["labor_income_mult_weighted"]).number_format = "0.0000"
    ws.cell(row=r, column=8, value=row["labor_income_mult_simple_avg"]).number_format = "0.0000"

# Column widths
for col, width in zip("ABCDEFGH", [26, 10, 22, 22, 22, 22, 26, 26]):
    ws.column_dimensions[col].width = width
ws.row_dimensions[5].height = 46

# Methodology note
note_row = 6 + len(industry_rows) + 2
notes = [
    "Methodology:",
    "  - Domestic intermediate flows Z[i,j] = PURCHASER - WHOLESALE - RETAIL - TRANSPORT - IMPORT (producer prices).",
    "  - Total output x[j] from row 210 \"Control total\" of the 2021 IO table.",
    "  - A = Z / x (column-normalized), Leontief inverse L = (I - A)^-1.",
    "  - Type I output multiplier for sector j = sum of column j of L (direct + indirect output generated per baht of final demand).",
    "  - Labor coefficient h_i = wages_i / x_i, using row 201 \"Wages and salaries\" as the labor-compensation proxy.",
    "  - Type I labor-income multiplier for sector j = (h'L)[j] / h[j] = (direct + indirect labor income) / direct labor income.",
    "  - Industry multipliers are output-weighted averages of member sectors' multipliers. A simple average is also shown.",
    "",
    "Important caveat on employment:",
    "  - A true employment multiplier requires employment (headcount or full-time-equivalent) by sector.",
    "    The IO file does not include employment data, so we report a labor-income (wage-based) multiplier as the",
    "    closest available proxy. Provide sector-level employment data to compute a physical employment multiplier.",
]
for k, line in enumerate(notes):
    c = ws.cell(row=note_row + k, column=1, value=line)
    if line.startswith(("Methodology:", "Important caveat on employment:")):
        c.font = bold
    ws.merge_cells(start_row=note_row + k, start_column=1, end_row=note_row + k, end_column=8)

# --- Sector-level detail sheet ---
ws2 = out.create_sheet("Sector detail")
headers2 = [
    "Code",
    "Sector",
    "Industry of the future",
    "Total output (million baht)",
    "Wages & salaries (million baht)",
    "Labor coefficient (wages/output)",
    "Output multiplier (Type I)",
    "Labor-income multiplier (Type I)",
]
for j, h_ in enumerate(headers2, start=1):
    c = ws2.cell(row=1, column=j, value=h_)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
ws2.row_dimensions[1].height = 40

for i, code in enumerate(sector_codes):
    r = i + 2
    ws2.cell(row=r, column=1, value=code)
    ws2.cell(row=r, column=2, value=sector_name[code])
    ws2.cell(row=r, column=3, value=industry_of.get(code, ""))
    ws2.cell(row=r, column=4, value=float(x[i])).number_format = "#,##0"
    ws2.cell(row=r, column=5, value=float(w[i])).number_format = "#,##0"
    ws2.cell(row=r, column=6, value=float(h[i])).number_format = "0.0000"
    ws2.cell(row=r, column=7, value=float(output_mult[i])).number_format = "0.0000"
    lim_i = labor_income_mult_typeI[i]
    ws2.cell(
        row=r,
        column=8,
        value=float(lim_i) if np.isfinite(lim_i) else None,
    ).number_format = "0.0000"

for col, width in zip("ABCDEFGH", [8, 46, 24, 22, 22, 22, 22, 24]):
    ws2.column_dimensions[col].width = width
ws2.freeze_panes = "C2"

# --- A matrix and Leontief inverse sheets ---
def write_matrix(sheet_name, M):
    s = out.create_sheet(sheet_name)
    s.cell(row=1, column=1, value="Code \\ Code").font = bold
    for j, code in enumerate(sector_codes, start=2):
        s.cell(row=1, column=j, value=code).font = bold
        s.cell(row=j, column=1, value=code).font = bold
    for i in range(N):
        for j in range(N):
            v = float(M[i, j])
            if v != 0:
                s.cell(row=i + 2, column=j + 2, value=v).number_format = "0.0000"
    s.freeze_panes = "B2"
    s.column_dimensions["A"].width = 10

write_matrix("A matrix", A)
write_matrix("Leontief inverse", L)

out.save(OUT)

# --- Print a concise console summary ---
print("\nIndustry-of-the-future multipliers (Type I):\n")
print(f"{'Industry':26s} {'n':>4s} {'Output mult.':>14s} {'Labor-income mult.':>20s}")
for r in industry_rows:
    print(
        f"{r['industry']:26s} {r['n_sectors']:>4d} "
        f"{r['output_mult_weighted']:>14.4f} {r['labor_income_mult_weighted']:>20.4f}"
    )
print(f"\nSaved: {OUT}")
