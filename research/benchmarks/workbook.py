from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from .common import ROOT, RUN_ROOT, file_hash, read_json, timestamp, write_json


def node_executable():
    override = os.environ.get("FAULTCRAFTER_NODE")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    executable = override or shutil.which("node") or (str(bundled) if bundled.exists() else None)
    if not executable:
        raise RuntimeError("Automatic Excel reports require Node.js 22+: install node or set FAULTCRAFTER_NODE")
    return executable


def column_letter(index):
    label = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def portable_xlsx(specifications, destination):
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relations = "http://schemas.openxmlformats.org/package/2006/relationships"
    documents = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
    content = "http://schemas.openxmlformats.org/package/2006/content-types"
    workbook = Element("workbook", xmlns=namespace)
    sheets = SubElement(workbook, "sheets")
    links = Element("Relationships", xmlns=relations)
    types = Element("Types", xmlns=content)
    for extension, mime in (("rels", "application/vnd.openxmlformats-package.relationships+xml"), ("xml", "application/xml")):
        SubElement(types, "Default", Extension=extension, ContentType=mime)
    styles = Element("styleSheet", xmlns=namespace)
    formats = sorted({column["format"] for sheet in specifications for column in sheet["columns"] if column["format"]})
    num_formats = SubElement(styles, "numFmts", count=str(len(formats)))
    for index, value in enumerate(formats, 164):
        SubElement(num_formats, "numFmt", numFmtId=str(index), formatCode=value)
    fonts = SubElement(styles, "fonts", count="2")
    for white in (False, True):
        font = SubElement(fonts, "font")
        SubElement(font, "sz", val="10")
        SubElement(font, "name", val="Microsoft YaHei")
        SubElement(font, "color", rgb="FFFFFFFF" if white else "FF243447")
        if white:
            SubElement(font, "b")
    fills = SubElement(styles, "fills", count="3")
    for value in ("none", "gray125", "solid"):
        pattern = SubElement(SubElement(fills, "fill"), "patternFill", patternType=value)
        if value == "solid":
            SubElement(pattern, "fgColor", rgb="FF234C72")
    borders = SubElement(styles, "borders", count="1")
    SubElement(borders, "border")
    style_xfs = SubElement(styles, "cellStyleXfs", count="1")
    SubElement(style_xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0")
    xfs = SubElement(styles, "cellXfs", count=str(len(formats) + 3))
    SubElement(xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0", xfId="0")
    header = SubElement(xfs, "xf", numFmtId="0", fontId="1", fillId="2", borderId="0", xfId="0", applyAlignment="1")
    SubElement(header, "alignment", wrapText="1", vertical="center", horizontal="center")
    for index in range(len(formats)):
        SubElement(xfs, "xf", numFmtId=str(164 + index), fontId="0", fillId="0", borderId="0", xfId="0", applyNumberFormat="1")
    wrapped = SubElement(xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0", xfId="0", applyAlignment="1")
    SubElement(wrapped, "alignment", wrapText="1", vertical="center")
    cell_styles = SubElement(styles, "cellStyles", count="1")
    SubElement(cell_styles, "cellStyle", name="Normal", xfId="0", builtinId="0")
    table_styles = SubElement(styles, "tableStyles", count="0", defaultTableStyle="TableStyleMedium2", defaultPivotStyle="PivotStyleLight16")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".xlsx.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def write(name, root):
            archive.writestr(name, tostring(root, encoding="utf-8", xml_declaration=True))

        for index, specification in enumerate(specifications, 1):
            sheet_id = f"rId{index}"
            SubElement(sheets, "sheet", name=specification["name"], sheetId=str(index), attrib={"xmlns:r": documents.rstrip("/"), "r:id": sheet_id})
            SubElement(links, "Relationship", Id=sheet_id, Type=documents + "worksheet", Target=f"worksheets/sheet{index}.xml")
            SubElement(types, "Override", PartName=f"/xl/worksheets/sheet{index}.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
            columns, rows = specification["columns"], specification["rows"]
            last = f"{column_letter(len(columns))}{len(rows) + 1}"
            worksheet = Element("worksheet", xmlns=namespace)
            views = SubElement(worksheet, "sheetViews")
            view = SubElement(views, "sheetView", workbookViewId="0", showGridLines="0")
            frozen_columns = 3 if index == 2 else 2
            SubElement(view, "pane", xSplit=str(frozen_columns), ySplit="1", topLeftCell=column_letter(frozen_columns + 1) + "2",
                       activePane="bottomRight", state="frozen")
            widths = SubElement(worksheet, "cols")
            for position, column in enumerate(columns, 1):
                SubElement(widths, "col", min=str(position), max=str(position), width=str(column["width"]), customWidth="1")
            data = SubElement(worksheet, "sheetData")
            matrix = [[column["title"] for column in columns], *[[row.get(column["key"]) for column in columns] for row in rows]]
            for row_index, values in enumerate(matrix, 1):
                lines = max([1, *[str(value).count("\n") + 1 for value in values if value is not None]])
                row = SubElement(data, "row", r=str(row_index), ht="42" if row_index == 1 else str(min(409, max(23, lines * 16 + 6))), customHeight="1")
                for position, value in enumerate(values, 1):
                    if value is None:
                        continue
                    cell = SubElement(row, "c", r=f"{column_letter(position)}{row_index}")
                    number_format = columns[position - 1]["format"]
                    if row_index == 1:
                        cell.set("s", "1")
                    elif number_format:
                        cell.set("s", str(2 + formats.index(number_format)))
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        SubElement(cell, "v").text = str(value)
                    else:
                        text_value = str(value)
                        if len(text_value) > 32767:
                            raise ValueError("Excel text exceeds the cell size limit")
                        cell.set("t", "inlineStr")
                        if row_index > 1 and columns[position - 1]["key"] in (
                                "training_devices", "training_channels", "training_conditions", "training_states",
                                "components", "feature_projection", "precision", "auxiliary_phase_updates"):
                            cell.set("s", str(2 + len(formats)))
                        SubElement(SubElement(cell, "is"), "t").text = text_value
            SubElement(worksheet, "autoFilter", ref="A1:" + last)
            write(f"xl/worksheets/sheet{index}.xml", worksheet)
        SubElement(links, "Relationship", Id="rIdStyles", Type=documents + "styles", Target="styles.xml")
        SubElement(types, "Override", PartName="/xl/workbook.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
        SubElement(types, "Override", PartName="/xl/styles.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml")
        root_links = Element("Relationships", xmlns=relations)
        SubElement(root_links, "Relationship", Id="rId1", Type=documents + "officeDocument", Target="xl/workbook.xml")
        write("_rels/.rels", root_links)
        write("xl/_rels/workbook.xml.rels", links)
        write("xl/workbook.xml", workbook)
        write("xl/styles.xml", styles)
        write("[Content_Types].xml", types)
    os.replace(temporary, destination)


def generate_workbook(run, data_run, *, output=None, artifact_runtime=None):
    if not (run / "diagnostic_groups.json").exists():
        from .diagnostic_groups import evaluate

        evaluate(run)
    if not (run / "quality_summary.json").exists():
        from .quality import evaluate

        evaluate(data_run, run, device="cpu", completed_only=True)
    destination = output or run / "comparison_experiments.xlsx"
    command = [node_executable(), str(Path(__file__).with_name("experiment_workbook.mjs")),
               "--run", str(run), "--data-run", str(data_run), "--output", str(destination)]
    specification_path = run / "workbook_data.json"
    subprocess.run([*command, "--spec-only", str(specification_path)], check=True)
    specifications = read_json(specification_path)
    for specification in specifications:
        keys = [column["key"] for column in specification["columns"]]
        if len(keys) != len(set(keys)) or "seed" not in keys:
            raise ValueError("Excel columns must have unique keys and a separate seed field")
        for row in specification["rows"]:
            if row.get("seed") not in (42, 43, 44, 45, 46):
                raise ValueError("Every Excel row must identify its experiment seed")
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError("Nonfinite Excel results must be represented by explicit missing values")
    if artifact_runtime:
        subprocess.run([*command, "--runtime", str(artifact_runtime)], check=True)
    else:
        portable_xlsx(specifications, destination)
    write_json(run / "workbook.json", {"created": timestamp(), "path": str(destination), "sha256": file_hash(destination),
               "data_sha256": file_hash(specification_path), "sheets": [{"name": sheet["name"], "rows": len(sheet["rows"]),
               "columns": len(sheet["columns"])} for sheet in specifications], "seed_column": "seed"})
    print(f"Excel report: {destination}; sheets={len(specifications)}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Regenerate the detailed, seeded Excel report from experiment records.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-runtime", type=Path)
    arguments = parser.parse_args()
    generate_workbook(arguments.run.resolve(), arguments.data_run.resolve(), output=arguments.output,
                      artifact_runtime=arguments.artifact_runtime)


if __name__ == "__main__":
    main()
