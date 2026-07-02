from __future__ import annotations

from datetime import datetime


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 11 Tf", "50 790 Td"]
    for line in lines:
        content_lines.append(f"({_pdf_escape(line)}) Tj")
        content_lines.append("0 -16 Td")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )
    objects.append(
        f"5 0 obj << /Length {len(content_stream)} >> stream\n".encode("ascii")
        + content_stream
        + b"\nendstream endobj\n"
    )

    header = b"%PDF-1.4\n"
    offsets = []
    body = b""
    cursor = len(header)
    for obj in objects:
        offsets.append(cursor)
        body += obj
        cursor += len(obj)

    xref_start = cursor
    xref = [b"xref\n", f"0 {len(objects)+1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer << /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF"
    )
    return header + body + b"".join(xref) + trailer


def generate_pdf_report(
    case_id: str,
    diagnosis: dict,
    report_agent1: str,
    symptoms: str,
    active_zone: str,
) -> bytes:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "MedAI Vision - AI Powered Clinico-Radiological Diagnosis Report",
        f"Generated: {now}",
        f"Case ID: {case_id}",
        "",
        "Agent 1 - Vision Report:",
    ]
    lines.extend(report_agent1.splitlines()[:8])
    lines.extend(
        [
            "",
            f"Grad-CAM active zone: {active_zone}",
            "",
            "Clinical Inputs:",
            symptoms[:180],
            "",
            "Agent 2 - Diagnostic Output:",
            f"Label: {diagnosis.get('label', '--')}",
            f"Pathology: {diagnosis.get('pathology', '--')}",
            f"Confidence: {diagnosis.get('confidence', 0):.1%}",
            f"Severity: {diagnosis.get('severity', '--')}",
            f"Concordance: {diagnosis.get('concordance', '--')}",
            f"Decision: {diagnosis.get('decision', '--')}",
            "",
            "Recommendation:",
            diagnosis.get("recommendation", "--")[:180],
            "",
            "Warning:",
            diagnosis.get("warning", "--")[:180],
        ]
    )
    return _build_pdf(lines)
