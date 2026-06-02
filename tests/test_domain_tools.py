import pathlib
import tempfile
import zipfile


def _registry(tmp: pathlib.Path, role: str = "user"):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=pathlib.Path.cwd(), drive_root=tmp)
    reg.set_context(ToolContext(
        repo_dir=pathlib.Path.cwd(),
        drive_root=tmp,
        current_user_id=123,
        user_role=role,
    ))
    return reg


def _write_minimal_xlsx(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Model" sheetId="1" r:id="rId1"/></sheets>
</workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""")
        zf.writestr("xl/sharedStrings.xml", """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
  <si><t>DSCR</t></si><si><t>ICR</t></si>
</sst>""")
        zf.writestr("xl/worksheets/sheet1.xml", """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>1.25</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><f>B1*2</f><v>2.50</v></c></row>
  </sheetData>
</worksheet>""")


def test_domain_tools_are_user_safe_non_core():
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(pathlib.Path(tmp), role="user")
        tools = set(reg.available_tools())
        assert "hr_candidate_screen" in tools
        assert "credit_deck_challenge" in tools

        core_tools = {schema["function"]["name"] for schema in reg.schemas(core_only=True)}
        assert "hr_candidate_screen" not in core_tools
        assert "credit_deck_challenge" not in core_tools


def test_hr_candidate_screen_inline_guardrails():
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(pathlib.Path(tmp), role="user")
        result = reg.execute("hr_candidate_screen", {
            "role_text": "Нужен аналитик с Excel, SQL и аккуратной коммуникацией.",
            "candidate_text": "Кандидат делал отчеты в Excel и SQL.",
        })
        assert "HR Candidate Screen" in result
        assert "protected attributes" in result
        assert "Evidence matrix" in result


def test_credit_pack_check_flags_collateral_without_cash_flow():
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(pathlib.Path(tmp), role="user")
        result = reg.execute("credit_pack_check", {
            "deal_text": "Сделка безопасна: залог недвижимости, LTV 55%.",
        })
        assert "Credit Pack Check" in result
        assert "ORANGE / collateral" in result
        assert "cash-flow repayment evidence" in result


def test_analyze_document_extracts_xlsx_values_and_formulas():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _write_minimal_xlsx(root / "model.xlsx")
        reg = _registry(root, role="user")
        result = reg.execute("analyze_document", {
            "path": "model.xlsx",
            "max_chars": 20_000,
        })
        assert "- type: xlsx" in result
        assert "A1=DSCR" in result
        assert "B2==B1*2 -> 2.50" in result


def test_domain_tool_rejects_drive_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(pathlib.Path(tmp), role="user")
        result = reg.execute("hr_vacancy_audit", {
            "vacancy_path": "../outside.txt",
        })
        assert "TOOL_ERROR" in result or "Path traversal" in result
