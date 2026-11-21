"""Low-level helpers that append content to an existing .docx in the formatting of BAO_CAO_ATTT.docx.

Formatting is copied from the original report: body paragraphs (line 1.5, first-line indent 1.27 cm),
"-" bullets (numId 5), manual heading numbers, SEQ-numbered captions (tables above, figures below),
TableGrid tables 16 cm wide with a grey header row, and code blocks as one shaded cell.
"""
from __future__ import annotations

import re
from copy import deepcopy

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

TEXT_WIDTH = 9072  # twips (A4, margins 1701 / 1134)


def _el(tag: str, **attrs) -> OxmlElement:
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def _sub(parent, tag: str, **attrs):
    e = _el(tag, **attrs)
    parent.append(e)
    return e


class Writer:
    def __init__(self, doc):
        self.doc = doc
        self.body = doc.element.body
        self.sect = self.body.sectPr

    # ---------------------------------------------------------------- basics
    def _append(self, el):
        self.sect.addprevious(el)
        return el

    @staticmethod
    def runs(p, text: str, size: int | None = None, bold: bool = False, font: str | None = None,
             italic: bool = False):
        """Inline markup: **bold**, *italic*, `mono`."""
        for part in re.split(r"(\*\*.+?\*\*|`.+?`|\*[^*\s][^*]*?\*)", text):
            if not part:
                continue
            b, i, mono = bold, italic, False
            if part.startswith("**") and part.endswith("**"):
                part, b = part[2:-2], True
            elif part.startswith("`") and part.endswith("`"):
                part, mono = part[1:-1], True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                part, i = part[1:-1], True
            r = _sub(p, "w:r")
            rpr = _sub(r, "w:rPr")
            if mono or font:
                f = "Consolas" if mono else font
                _sub(rpr, "w:rFonts", **{"w:ascii": f, "w:hAnsi": f, "w:cs": f})
            if b:
                _sub(rpr, "w:b"); _sub(rpr, "w:bCs")
            if i:
                _sub(rpr, "w:i"); _sub(rpr, "w:iCs")
            sz = size if size else (None if not mono else None)
            if mono and not size:
                sz = 22
            if sz:
                _sub(rpr, "w:sz", **{"w:val": sz}); _sub(rpr, "w:szCs", **{"w:val": sz})
            if len(rpr) == 0:
                r.remove(rpr)
            t = _sub(r, "w:t")
            t.text = part
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p

    def _p(self, style: str | None = None):
        p = _el("w:p")
        ppr = _sub(p, "w:pPr")
        if style:
            _sub(ppr, "w:pStyle", **{"w:val": style})
        return p, ppr

    # ---------------------------------------------------------------- blocks
    def h1(self, text: str):
        p, ppr = self._p("Heading1")
        _sub(ppr, "w:pageBreakBefore")
        num = _sub(ppr, "w:numPr"); _sub(num, "w:ilvl", **{"w:val": 0}); _sub(num, "w:numId", **{"w:val": 0})
        _sub(ppr, "w:ind", **{"w:left": 432, "w:hanging": 432})
        _sub(ppr, "w:jc", **{"w:val": "left"})
        return self._append(self.runs(p, text))

    def h2(self, text: str):
        p, ppr = self._p("Heading2")
        _sub(ppr, "w:keepNext")
        _sub(ppr, "w:spacing", **{"w:before": 160})
        return self._append(self.runs(p, text))

    def h3(self, text: str):
        p, ppr = self._p("Heading3")
        _sub(ppr, "w:keepNext")
        num = _sub(ppr, "w:numPr"); _sub(num, "w:ilvl", **{"w:val": 0}); _sub(num, "w:numId", **{"w:val": 0})
        _sub(ppr, "w:spacing", **{"w:before": 120, "w:line": 360, "w:lineRule": "auto"})
        _sub(ppr, "w:ind", **{"w:left": 720, "w:hanging": 720})
        return self._append(self.runs(p, text))

    def para(self, text: str, indent: bool = True, keep_next: bool = False, italic: bool = False):
        p, ppr = self._p()
        if keep_next:
            _sub(ppr, "w:keepNext")
        _sub(ppr, "w:spacing", **{"w:line": 360, "w:lineRule": "auto"})
        if indent:
            _sub(ppr, "w:ind", **{"w:firstLine": 720})
        return self._append(self.runs(p, text, italic=italic))

    def bullets(self, items: list[str]):
        for it in items:
            p, ppr = self._p("ListParagraph")
            num = _sub(ppr, "w:numPr"); _sub(num, "w:ilvl", **{"w:val": 0}); _sub(num, "w:numId", **{"w:val": 5})
            _sub(ppr, "w:spacing", **{"w:line": 360, "w:lineRule": "auto"})
            _sub(ppr, "w:ind", **{"w:left": 360})
            self._append(self.runs(p, it))

    def subhead(self, text: str):
        """Bold run-in paragraph (appendix sub-items like 'A.1.')."""
        p, ppr = self._p("NoSpacing")
        _sub(ppr, "w:keepNext")
        _sub(ppr, "w:spacing", **{"w:before": 120, "w:line": 360, "w:lineRule": "auto"})
        return self._append(self.runs(p, text, bold=True))

    def caption(self, kind: str, text: str, keep_next: bool):
        p, ppr = self._p("Caption")
        if keep_next:
            _sub(ppr, "w:keepNext")
        _sub(ppr, "w:spacing", **{"w:before": 120, "w:line": 360, "w:lineRule": "auto"})
        _sub(ppr, "w:jc", **{"w:val": "center"})
        self.runs(p, f"{kind} ")
        fld = _sub(p, "w:fldSimple", **{"w:instr": f" SEQ {kind} \\* ARABIC "})
        r = _sub(fld, "w:r"); rpr = _sub(r, "w:rPr"); _sub(rpr, "w:noProof")
        _sub(r, "w:t").text = "1"
        self.runs(p, f". {text}")
        return self._append(p)

    def figure(self, path, caption: str, width_cm: float = 15.5):
        par = self.doc.add_paragraph()            # inserted before sectPr, i.e. at the end
        par.add_run().add_picture(str(path), width=Cm(width_cm))
        ppr = par._p.get_or_add_pPr()
        ppr.append(_el("w:keepNext"))
        ppr.append(_el("w:spacing", **{"w:before": 120, "w:line": 240, "w:lineRule": "auto"}))
        ppr.append(_el("w:jc", **{"w:val": "center"}))
        self.caption("Hình", caption, keep_next=False)

    def table(self, caption: str, header: list[str], rows: list[list], widths: list[float],
              align: str | None = None, size: int = 22, bold_first_col: bool = False,
              highlight_rows: set[int] | None = None, note: str | None = None):
        """widths: relative; align: one char per column (l/c/r)."""
        self.caption("Bảng", caption, keep_next=True)
        total = sum(widths)
        tw = [int(TEXT_WIDTH * w / total) for w in widths]
        tw[-1] += TEXT_WIDTH - sum(tw)
        align = align or "l" * len(header)
        jc = {"l": "left", "c": "center", "r": "right"}
        tbl = _el("w:tbl")
        tpr = _sub(tbl, "w:tblPr")
        _sub(tpr, "w:tblStyle", **{"w:val": "TableGrid"})
        _sub(tpr, "w:tblW", **{"w:w": TEXT_WIDTH, "w:type": "dxa"})
        _sub(tpr, "w:jc", **{"w:val": "center"})
        _sub(tpr, "w:tblLayout", **{"w:type": "fixed"})
        mar = _sub(tpr, "w:tblCellMar")
        _sub(mar, "w:left", **{"w:w": 85, "w:type": "dxa"}); _sub(mar, "w:right", **{"w:w": 85, "w:type": "dxa"})
        _sub(tpr, "w:tblLook", **{"w:val": "04A0", "w:firstRow": 1, "w:lastRow": 0, "w:firstColumn": 1,
                                  "w:lastColumn": 0, "w:noHBand": 0, "w:noVBand": 1})
        grid = _sub(tbl, "w:tblGrid")
        for w in tw:
            _sub(grid, "w:gridCol", **{"w:w": w})
        all_rows = [header] + rows
        keep_together = len(all_rows) <= 16          # short tables never split across pages
        for ri, row in enumerate(all_rows):
            tr = _sub(tbl, "w:tr")
            trpr = _sub(tr, "w:trPr")
            _sub(trpr, "w:cantSplit")
            if ri == 0:
                _sub(trpr, "w:tblHeader")
            _sub(trpr, "w:jc", **{"w:val": "center"})
            for ci, val in enumerate(row):
                tc = _sub(tr, "w:tc")
                tcpr = _sub(tc, "w:tcPr")
                _sub(tcpr, "w:tcW", **{"w:w": tw[ci], "w:type": "dxa"})
                if ri == 0:
                    _sub(tcpr, "w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": "D9E2F3"})
                elif highlight_rows and ri in highlight_rows:
                    _sub(tcpr, "w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": "EAF2E3"})
                _sub(tcpr, "w:vAlign", **{"w:val": "center"})
                for li, line in enumerate(str(val).split("\n")):
                    p = _sub(tc, "w:p")
                    ppr = _sub(p, "w:pPr")
                    if ri == 0 or (keep_together and ri < len(all_rows) - 1):
                        _sub(ppr, "w:keepNext")
                    _sub(ppr, "w:spacing", **{"w:before": 30, "w:after": 30, "w:line": 252, "w:lineRule": "auto"})
                    _sub(ppr, "w:ind", **{"w:firstLine": 0})
                    _sub(ppr, "w:jc", **{"w:val": "center" if ri == 0 else jc[align[ci]]})
                    self.runs(p, line, size=size, bold=(ri == 0) or (bold_first_col and ci == 0))
        self._append(tbl)
        if note:
            p, ppr = self._p()
            _sub(ppr, "w:spacing", **{"w:before": 60, "w:line": 276, "w:lineRule": "auto"})
            self._append(self.runs(p, note, size=22, italic=True))
        else:
            self.spacer()

    def spacer(self):
        p, ppr = self._p()
        _sub(ppr, "w:spacing", **{"w:before": 0, "w:after": 0, "w:line": 120, "w:lineRule": "auto"})
        r = _sub(p, "w:r"); rpr = _sub(r, "w:rPr"); _sub(rpr, "w:sz", **{"w:val": 8})
        return self._append(p)

    def code(self, text: str):
        tbl = _el("w:tbl")
        tpr = _sub(tbl, "w:tblPr")
        _sub(tpr, "w:tblStyle", **{"w:val": "TableGrid"})
        _sub(tpr, "w:tblW", **{"w:w": TEXT_WIDTH, "w:type": "dxa"})
        _sub(tpr, "w:jc", **{"w:val": "center"})
        _sub(tpr, "w:tblLayout", **{"w:type": "fixed"})
        grid = _sub(tbl, "w:tblGrid"); _sub(grid, "w:gridCol", **{"w:w": TEXT_WIDTH})
        tr = _sub(tbl, "w:tr"); tc = _sub(tr, "w:tc"); tcpr = _sub(tc, "w:tcPr")
        _sub(tcpr, "w:tcW", **{"w:w": TEXT_WIDTH, "w:type": "dxa"})
        _sub(tcpr, "w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": "F5F5F5"})
        m = _sub(tcpr, "w:tcMar"); _sub(m, "w:top", **{"w:w": 80, "w:type": "dxa"})
        _sub(m, "w:bottom", **{"w:w": 80, "w:type": "dxa"})
        for line in text.strip("\n").split("\n"):
            p = _sub(tc, "w:p"); ppr = _sub(p, "w:pPr")
            _sub(ppr, "w:spacing", **{"w:before": 0, "w:after": 0, "w:line": 240, "w:lineRule": "auto"})
            _sub(ppr, "w:ind", **{"w:firstLine": 0})
            _sub(ppr, "w:jc", **{"w:val": "left"})
            r = _sub(p, "w:r"); rpr = _sub(r, "w:rPr")
            _sub(rpr, "w:rFonts", **{"w:ascii": "Consolas", "w:hAnsi": "Consolas", "w:cs": "Consolas"})
            _sub(rpr, "w:noProof")
            _sub(rpr, "w:sz", **{"w:val": 17}); _sub(rpr, "w:szCs", **{"w:val": 17})
            t = _sub(r, "w:t"); t.text = line
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        self._append(tbl)
        self.spacer()


def set_paragraph_text(p_el, text: str, writer: Writer):
    """Replace a paragraph's runs, keeping its pPr."""
    for child in list(p_el):
        if child.tag != qn("w:pPr"):
            p_el.remove(child)
    writer.runs(p_el, text)


def toc_field(p_el, instr: str = ' TOC \\o "1-3" \\h \\z \\u '):
    for child in list(p_el):
        if child.tag != qn("w:pPr"):
            p_el.remove(child)
    for kind, extra in (("begin", None), ("instr", instr), ("separate", None), ("text", "Cập nhật mục lục: chuột phải → Update Field."), ("end", None)):
        r = _sub(p_el, "w:r")
        if kind == "instr":
            t = _sub(r, "w:instrText"); t.text = extra
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        elif kind == "text":
            _sub(r, "w:t").text = extra
        else:
            _sub(r, "w:fldChar", **{"w:fldCharType": kind})


def clone_row(tbl, values: list[str]):
    last = tbl.rows[-1]._tr
    new = deepcopy(last)
    last.addnext(new)
    cells = new.findall(qn("w:tc"))
    for tc, val in zip(cells, values):
        ps = tc.findall(qn("w:p"))
        for extra in ps[1:]:
            tc.remove(extra)
        p = ps[0]
        rpr = None
        r0 = p.find(qn("w:r"))
        if r0 is not None and r0.find(qn("w:rPr")) is not None:
            rpr = deepcopy(r0.find(qn("w:rPr")))
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        r = _sub(p, "w:r")
        if rpr is not None:
            r.append(rpr)
        t = _sub(r, "w:t"); t.text = val
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
