#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import BDBRA-style CSV into WholeBIF-RDB (PostgreSQL). v2
- Fix: strictly cap reference_id at 10 chars
- Fix: optional truncation for VARCHAR(255) columns (--no-truncate to disable)
- CSV encoding options: --encoding (default utf-8), --errors (default replace)
"""

import argparse
import csv
import sys
from typing import Dict, Any, Optional, Tuple
from urllib.parse import quote
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except Exception as e:
    print("[ERROR] psycopg2 is required. Install with: pip install psycopg2-binary", file=sys.stderr)
    raise

# ===== Default DB settings (can be overridden by CLI args) =====
DEFAULTS = dict(
    host="localhost",
    port="5432",
    dbname="wholebif_rdb",
    user="wholebif",
    password="Ashi12137",
)

# Columns that are VARCHAR(255) in schema and may need truncation
VARCHAR255_COLUMNS = {
    # references_tbl
    "reference_id", "doc_link", "bibtex_link", "doi", "litterature_type",
    "type", "journal_names", "contributor", "project_id", "reviewer",
    # connections: none are varchar(255) except taxon/method maybe, but not in this set
}

def norm(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float):
        if s.is_integer():
            return str(int(s))
        return str(s)
    s = str(s).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def first_nonempty(*vals: Any) -> str:
    for v in vals:
        sv = norm(v)
        if sv:
            return sv
    return ""

def make_doc_link_from_doi(doi: str) -> str:
    doi = norm(doi)
    if not doi:
        return ""
    return f"https://doi.org/{doi}"

def make_bibtex_dataurl(bibtex: str) -> str:
    b = norm(bibtex)
    if not b:
        return ""
    return "data:text/plain;charset=utf-8," + quote(b)

def sanitize_id(s: str) -> str:
    return norm(s)

def gen_reference_id(reference_text: str, fallback: str = "") -> str:
    ref = norm(reference_text)
    if ref:
        return ref[:10]
    fb = norm(fallback)
    if fb:
        return fb[:10]
    return ("GEN" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))[:10]

def open_conn(args):
    return psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname,
        user=args.user, password=args.password
    )

def maybe_truncate_255(d: Dict[str, Any], enable: bool) -> Dict[str, Any]:
    if not enable:
        return d
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = v
            continue
        s = str(v)
        if k in VARCHAR255_COLUMNS and len(s) > 255:
            out[k] = s[:255]
        else:
            out[k] = v
    return out

def ensure_references(conn, ref_row: Dict[str, Any], truncate255: bool = True) -> str:
    ref_row = maybe_truncate_255(ref_row, truncate255)
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            INSERT INTO references_tbl (
                reference_id, doc_link, bibtex_link, doi, bibtex,
                litterature_type, type, authors, title, journal_names,
                alternative_url, contributor, project_id, review_results, reviewer
            ) VALUES (
                %(reference_id)s, %(doc_link)s, %(bibtex_link)s, %(doi)s, %(bibtex)s,
                %(litterature_type)s, %(type)s, %(authors)s, %(title)s, %(journal_names)s,
                %(alternative_url)s, %(contributor)s, %(project_id)s, %(review_results)s, %(reviewer)s
            )
            ON CONFLICT (reference_id) DO UPDATE SET
                doc_link = EXCLUDED.doc_link,
                bibtex_link = EXCLUDED.bibtex_link,
                doi = EXCLUDED.doi,
                bibtex = EXCLUDED.bibtex,
                litterature_type = COALESCE(EXCLUDED.litterature_type, references_tbl.litterature_type),
                type = COALESCE(EXCLUDED.type, references_tbl.type),
                authors = COALESCE(EXCLUDED.authors, references_tbl.authors),
                title = COALESCE(EXCLUDED.title, references_tbl.title),
                journal_names = COALESCE(EXCLUDED.journal_names, references_tbl.journal_names),
                alternative_url = COALESCE(EXCLUDED.alternative_url, references_tbl.alternative_url),
                contributor = EXCLUDED.contributor,
                project_id = COALESCE(EXCLUDED.project_id, references_tbl.project_id),
                review_results = COALESCE(EXCLUDED.review_results, references_tbl.review_results),
                reviewer = COALESCE(EXCLUDED.reviewer, references_tbl.reviewer)
            """,
            ref_row,
        )
    return ref_row["reference_id"]

def insert_connection(conn, con_row: Dict[str, Any]) -> bool:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            INSERT INTO connections (
                sender_circuit_id, receiver_circuit_id, reference_id,
                taxon, measurement_method, pointers_on_literature,
                pointers_on_figure, credibility_rating, summarized_cr, reviewer
            ) VALUES (
                %(sender_circuit_id)s, %(receiver_circuit_id)s, %(reference_id)s,
                %(taxon)s, %(measurement_method)s, %(pointers_on_literature)s,
                %(pointers_on_figure)s, %(credibility_rating)s, %(summarized_cr)s, %(reviewer)s
            )
            ON CONFLICT (sender_circuit_id, receiver_circuit_id, reference_id) DO NOTHING
            """,
            con_row
        )
        return True

def row_to_lowerkey(d: Dict[str, Any]) -> Dict[str, Any]:
    return { (k.lower().strip() if isinstance(k, str) else k): v for k, v in d.items() }

def build_reference_row(ld: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reference_text = first_nonempty(ld.get("reference"), ld.get("ref"), ld.get("citation"))
    doi = first_nonempty(ld.get("doi"), ld.get("dois"))
    bibtex = first_nonempty(ld.get("bibtex"), ld.get("bibtex_text"), ld.get("bib"))
    journal = first_nonempty(ld.get("journal"), ld.get("journal_name"), ld.get("journal_names"))
    title = first_nonempty(ld.get("title"), ld.get("paper_title"))
    authors = first_nonempty(ld.get("authors"), ld.get("author"))
    litterature_type = first_nonempty(ld.get("litterature_type"), ld.get("literature_type"), ld.get("type"))
    typ = first_nonempty(ld.get("type")) if not litterature_type else ""
    alternative_url = first_nonempty(ld.get("alternative_url"), ld.get("url"))
    project_id = norm(ld.get("project_id"))
    review_results = norm(ld.get("review_results"))
    reviewer = norm(ld.get("reviewer"))

    reference_id = gen_reference_id(reference_text, fallback=doi)[:10]
    doc_link = make_doc_link_from_doi(doi)
    bibtex_link = make_bibtex_dataurl(bibtex)

    ref_row = dict(
        reference_id = reference_id,
        doc_link = doc_link,
        bibtex_link = bibtex_link,
        doi = doi,
        bibtex = bibtex,
        litterature_type = litterature_type,
        type = typ,
        authors = authors,
        title = title,
        journal_names = journal,
        alternative_url = alternative_url,
        contributor = "fromBDBRA",
        project_id = project_id or None,
        review_results = review_results or None,
        reviewer = reviewer or None,
    )
    return reference_id, ref_row

def build_connection_row(ld: Dict[str, Any], reference_id: str) -> Optional[Dict[str, Any]]:
    sender = sanitize_id(first_nonempty(ld.get("dhbasid"), ld.get("sender_circuit_id"), ld.get("sender")))
    receiver = sanitize_id(first_nonempty(ld.get("dhbarid"), ld.get("receiver_circuit_id"), ld.get("receiver")))
    if not sender or not receiver:
        return None

    taxon = norm(ld.get("taxon"))
    method = first_nonempty(ld.get("method"), ld.get("measurement_method"))
    pointer = first_nonempty(ld.get("pointer"), ld.get("pointers_on_literature"), ld.get("evidence"), ld.get("pointers"))
    figure = first_nonempty(ld.get("figure"), ld.get("pointers_on_figure"), ld.get("fig"))
    credibility_rating = ld.get("credibility_rating")
    summarized_cr = ld.get("summarized_cr")
    reviewer = norm(ld.get("reviewer"))

    def to_float(x):
        try:
            xs = norm(x)
            return float(xs) if xs else None
        except Exception:
            return None

    con_row = dict(
        sender_circuit_id = sender,
        receiver_circuit_id = receiver,
        reference_id = reference_id,
        taxon = taxon or None,
        measurement_method = method or None,
        pointers_on_literature = pointer or None,
        pointers_on_figure = figure or None,
        credibility_rating = to_float(credibility_rating),
        summarized_cr = to_float(summarized_cr),
        reviewer = reviewer or None,
    )
    return con_row

def main():
    ap = argparse.ArgumentParser(description="Import CSV into WholeBIF-RDB (references_tbl + connections). v2")
    ap.add_argument("--csv", required=True, help="Path to CSV to import.")
    ap.add_argument("--host", default=DEFAULTS["host"])
    ap.add_argument("--port", default=DEFAULTS["port"])
    ap.add_argument("--dbname", default=DEFAULTS["dbname"])
    ap.add_argument("--user", default=DEFAULTS["user"])
    ap.add_argument("--password", default=DEFAULTS["password"])
    ap.add_argument("--commit_every", type=int, default=500, help="Commit interval for batch inserts.")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (utf-8/cp932/latin1 etc.)")
    ap.add_argument("--errors", default="replace", help="CSV decode errors policy (strict/ignore/replace)")
    ap.add_argument("--no-truncate", action="store_true", help="Do not truncate VARCHAR(255) fields (may cause errors)")
    args = ap.parse_args()

    conn = open_conn(args)
    conn.autocommit = False

    total = ok_refs = ok_conns = skipped_conns = 0
    batch = 0

    with open(args.csv, "r", newline="", encoding=args.encoding, errors=args.errors) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            ld = row_to_lowerkey(row)

            reference_id, ref_row = build_reference_row(ld)
            try:
                ensure_references(conn, ref_row, truncate255=(not args.no_truncate))
                ok_refs += 1
            except Exception as e:
                print(f"[ERROR] references_tbl upsert failed at row {total} (reference_id={reference_id}): {e}", file=sys.stderr)
                conn.rollback()
                continue

            con_row = build_connection_row(ld, reference_id)
            if con_row is None:
                skipped_conns += 1
            else:
                try:
                    inserted = insert_connection(conn, con_row)
                    if inserted:
                        ok_conns += 1
                except Exception as e:
                    print(f"[ERROR] connections insert failed at row {total}: {e}", file=sys.stderr)
                    conn.rollback()
                    continue

            batch += 1
            if batch >= args.commit_every:
                conn.commit()
                batch = 0

    conn.commit()
    conn.close()
    print(f"[DONE] Processed rows: {total}")
    print(f"        references_tbl upserted: {ok_refs}")
    print(f"        connections inserted:   {ok_conns}")
    print(f"        connections skipped:    {skipped_conns} (missing dhbasid/dhbarid)")

if __name__ == "__main__":
    main()
