"""XLSX text extraction using openpyxl."""
from pathlib import Path


def extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # Build markdown table
        header = rows[0]
        col_names = [str(c) if c is not None else "" for c in header]
        lines = [f"**Sheet: {sheet_name}**", ""]
        lines.append("| " + " | ".join(col_names) + " |")
        lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
        for row in rows[1:]:
            cells = [str(c) if c is not None else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        sheets.append("\n".join(lines))
    wb.close()
    return "\n\n".join(sheets) if sheets else "[No data found in spreadsheet]"
