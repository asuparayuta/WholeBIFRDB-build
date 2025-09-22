#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradio_wholebif_query_app_flexpair_public.py  (self-contained, public-ready)
----------------------------------------------------------------------------
- Query Explorer（類似検索）
- Pair Lookup（Sender/Receiver を個別指定）
- Flex Pair Finder（1領域から柔軟に対向候補 → ペア確定）
- 公開用ランチャー機能（--share / --host / --port / --auth）

Examples
--------
# 一時公開（共有リンク発行、超手軽）
python gradio_wholebif_query_app_flexpair_public.py --share --auth user:pass

# LAN/インターネット公開（自前で待ち受け、ポート開放など必要）
python gradio_wholebif_query_app_flexpair_public.py --host 0.0.0.0 --port 7860 --auth user:pass

Env vars
--------
GRADIO_SHARE=1 / true / yes
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_AUTH='user1:pass1,user2:pass2'

Security
--------
- 公開時は必ず --auth または GRADIO_AUTH を設定してください。
- 本番運用では、逆プロキシ + TLS（Caddy/NGINX）を推奨します。
"""

import os
import json
import argparse
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import gradio as gr


# ------------------------------
# DB helpers & detection
# ------------------------------

@dataclass
class DBFlags:
    has_evidence: bool
    has_refs_view: bool
    refs_source: str             # "refs" or "references_tbl"
    has_scores: bool
    has_pg_trgm: bool
    has_connections_std: bool


def get_dsn() -> Dict[str, Any]:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db   = os.getenv("POSTGRES_DB", "wholebif_rdb")
    user = os.getenv("POSTGRES_USER", "wholebif")
    pwd  = os.getenv("POSTGRES_PASSWORD", "")
    return dict(host=host, port=port, dbname=db, user=user, password=pwd)


def detect_flags(conn) -> DBFlags:
    def _exists(kind: str, name: str) -> bool:
        if kind == "tables":
            qry = "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s"
        else:
            qry = "SELECT 1 FROM information_schema.views  WHERE table_schema='public' AND table_name=%s"
        with conn.cursor() as cur:
            cur.execute(qry, (name,))
            return cur.fetchone() is not None

    def _ext(name: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname=%s", (name,))
            return cur.fetchone() is not None

    has_evidence        = _exists("tables", "evidence")
    has_refs_view       = _exists("views", "refs")
    has_scores          = _exists("tables", "scores")
    has_pg_trgm         = _ext("pg_trgm")
    has_connections_std = _exists("views", "connections_std")
    refs_source         = "refs" if has_refs_view else ("references_tbl" if _exists("tables", "references_tbl") else "refs")

    return DBFlags(
        has_evidence=has_evidence,
        has_refs_view=has_refs_view,
        refs_source=refs_source,
        has_scores=has_scores,
        has_pg_trgm=has_pg_trgm,
        has_connections_std=has_connections_std,
    )


def refs_join_cols_for_source(refs_source: str) -> Tuple[str, str]:
    if refs_source == "refs":
        return ("LEFT JOIN refs r ON r.reference_id = x.reference_id",
                "r.reference_id, r.doi, r.title, r.journal, r.year, r.url")
    else:
        return ("LEFT JOIN references_tbl r ON r.reference_id = x.reference_id",
                "r.reference_id, r.doi, r.title, r.journal_names AS journal, NULL::int AS year, COALESCE(r.doc_link, r.alternative_url) AS url")


# ------------------------------
# Query builders & suggestors
# ------------------------------

def circuits_like_sql(flags: DBFlags) -> str:
    if flags.has_pg_trgm:
        return """
        SELECT circuit_id, names,
               GREATEST(similarity(circuit_id, %(q)s), similarity(COALESCE(names,''), %(q)s)) AS sim
        FROM circuits
        WHERE circuit_id %% %(q)s OR COALESCE(names,'') %% %(q)s
        ORDER BY sim DESC
        LIMIT %(limit)s;
        """
    else:
        return """
        SELECT circuit_id, names, NULL::float AS sim
        FROM circuits
        WHERE circuit_id ILIKE %(pat)s OR COALESCE(names,'') ILIKE %(pat)s
        ORDER BY circuit_id
        LIMIT %(limit)s;
        """


def receivers_like_sql(flags: DBFlags) -> str:
    if flags.has_pg_trgm:
        return """
        SELECT DISTINCT receiver_circuit_id AS receiver_id,
               similarity(receiver_circuit_id, %(q)s) AS sim
        FROM connections
        WHERE receiver_circuit_id %% %(q)s
        ORDER BY sim DESC
        LIMIT %(limit)s;
        """
    else:
        return """
        SELECT DISTINCT receiver_circuit_id AS receiver_id, NULL::float AS sim
        FROM connections
        WHERE receiver_circuit_id ILIKE %(pat)s
        ORDER BY receiver_circuit_id
        LIMIT %(limit)s;
        """


def suggest_circuit_ids(partial: str, topn: int = 12):
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        with psycopg2.connect(**get_dsn()) as conn:
            flags = detect_flags(conn)
            with conn.cursor() as cur:
                if flags.has_pg_trgm:
                    cur.execute(
                        """
                        SELECT circuit_id
                        FROM circuits
                        WHERE circuit_id %% %(q)s OR COALESCE(names,'') %% %(q)s
                        ORDER BY GREATEST(similarity(circuit_id, %(q)s), similarity(COALESCE(names,''), %(q)s)) DESC
                        LIMIT %(limit)s
                        """,
                        {"q": q, "limit": topn}
                    )
                else:
                    cur.execute(
                        """
                        SELECT circuit_id
                        FROM circuits
                        WHERE circuit_id ILIKE %(pat)s OR COALESCE(names,'') ILIKE %(pat)s
                        ORDER BY circuit_id
                        LIMIT %(limit)s
                        """,
                        {"pat": f"%{q}%", "limit": topn}
                    )
                choices = [r[0] for r in cur.fetchall()]
        return gr.update(choices=choices, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def suggest_receiver_ids(partial: str, topn: int = 12):
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        with psycopg2.connect(**get_dsn()) as conn:
            flags = detect_flags(conn)
            with conn.cursor() as cur:
                if flags.has_pg_trgm:
                    cur.execute(
                        """
                        SELECT DISTINCT receiver_circuit_id AS receiver_id
                        FROM connections
                        WHERE receiver_circuit_id %% %(q)s
                        ORDER BY similarity(receiver_circuit_id, %(q)s) DESC
                        LIMIT %(limit)s
                        """,
                        {"q": q, "limit": topn}
                    )
                else:
                    cur.execute(
                        """
                        SELECT DISTINCT receiver_circuit_id AS receiver_id
                        FROM connections
                        WHERE receiver_circuit_id ILIKE %(pat)s
                        ORDER BY receiver_circuit_id
                        LIMIT %(limit)s
                        """,
                        {"pat": f"%{q}%", "limit": topn}
                    )
                choices = [r[0] for r in cur.fetchall()]
        return gr.update(choices=choices, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def suggest_any_region(partial: str, topn: int = 12):
    """circuits.circuit_id / circuits.names / connections.receiver_circuit_id を横断して候補提示"""
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        with psycopg2.connect(**get_dsn()) as conn:
            flags = detect_flags(conn)
            cur = conn.cursor()
            items: List[Tuple[str, float]] = []  # (id, score)

            if flags.has_pg_trgm:
                # circuits: id / names
                cur.execute("""
                    SELECT circuit_id,
                           GREATEST(similarity(circuit_id, %(q)s), similarity(COALESCE(names,''), %(q)s)) AS sim
                    FROM circuits
                    WHERE circuit_id %% %(q)s OR COALESCE(names,'') %% %(q)s
                    ORDER BY sim DESC LIMIT %(limit)s
                """, {"q": q, "limit": topn})
                items += [(r[0], float(r[1])) for r in cur.fetchall()]

                # receivers
                cur.execute("""
                    SELECT DISTINCT receiver_circuit_id AS id,
                           similarity(receiver_circuit_id, %(q)s) AS sim
                    FROM connections
                    WHERE receiver_circuit_id %% %(q)s
                    ORDER BY sim DESC LIMIT %(limit)s
                """, {"q": q, "limit": topn})
                items += [(r[0], float(r[1])) for r in cur.fetchall() if r[0] is not None]
            else:
                pat = f"%{q}%"
                cur.execute("""
                    SELECT circuit_id, 1.0 AS sim
                    FROM circuits
                    WHERE circuit_id ILIKE %(pat)s OR COALESCE(names,'') ILIKE %(pat)s
                    ORDER BY circuit_id LIMIT %(limit)s
                """, {"pat": pat, "limit": topn})
                items += [(r[0], 1.0) for r in cur.fetchall()]

                cur.execute("""
                    SELECT DISTINCT receiver_circuit_id AS id, 1.0 AS sim
                    FROM connections
                    WHERE receiver_circuit_id ILIKE %(pat)s
                    ORDER BY receiver_circuit_id LIMIT %(limit)s
                """, {"pat": pat, "limit": topn})
                items += [(r[0], 1.0) for r in cur.fetchall() if r[0] is not None]

            # unique by id, keep max score
            best: Dict[str, float] = {}
            for id_, s in items:
                if id_ is None: continue
                if id_ not in best or s > best[id_]:
                    best[id_] = s
            # sort by score desc then id asc
            choices = [k for k,_ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:topn]
            return gr.update(choices=choices, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def apply_selection_to_text(selected: str):
    return gr.update(value=(selected or ""))


# ------------------------------
# Main search (Query Explorer tab)
# ------------------------------

def run_query(query: str, limit: int = 20):
    q = (query or "").strip()
    empty = pd.DataFrame()
    if q == "":
        return (empty, empty, empty, empty, empty,
                empty, empty, empty, empty, empty,
                "クエリを入力してください。")
    try:
        with psycopg2.connect(**get_dsn()) as conn:
            flags = detect_flags(conn)
            # 1) circuits
            with conn.cursor() as cur:
                sql_cir = circuits_like_sql(flags)
                params = {"q": q, "limit": limit, "pat": f"%{q}%"}
                cur.execute(sql_cir, params)
                crow = cur.fetchall()
                ccols = [d.name for d in cur.description]
                df_circuits = pd.DataFrame(crow, columns=ccols)

            # 2) receivers
            with conn.cursor() as cur:
                sql_rec = receivers_like_sql(flags)
                params = {"q": q, "limit": limit, "pat": f"%{q}%"}
                cur.execute(sql_rec, params)
                rrow = cur.fetchall()
                rcols = [d.name for d in cur.description]
                df_receivers = pd.DataFrame(rrow, columns=rcols)

            circuit_ids: List[str] = df_circuits["circuit_id"].dropna().astype(str).tolist()
            receiver_ids: List[str] = df_receivers["receiver_id"].dropna().astype(str).tolist()

            # Circuit-side Connections
            if circuit_ids:
                with conn.cursor() as cur:
                    placeholders = ",".join([f"%s" for _ in circuit_ids])
                    cur.execute(f"""
                        SELECT
                          c.sender_circuit_id AS circuit_id,
                          c.receiver_circuit_id AS receiver_id,
                          c.reference_id,
                          c.measurement_method AS method,
                          c.taxon,
                          c.pointers_on_literature,
                          c.pointers_on_figure,
                          c.credibility_rating,
                          c.summarized_cr,
                          c.reviewer
                        FROM connections c
                        WHERE c.sender_circuit_id IN ({placeholders})
                        ORDER BY c.sender_circuit_id, c.receiver_circuit_id, c.reference_id
                    """, circuit_ids)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_conn_c = pd.DataFrame(rows, columns=cols)
            else:
                df_conn_c = pd.DataFrame(columns=[
                    "circuit_id","receiver_id","reference_id","method","taxon",
                    "pointers_on_literature","pointers_on_figure","credibility_rating","summarized_cr","reviewer"
                ])

            # References for circuit-side
            if not df_conn_c.empty:
                ref_ids = df_conn_c["reference_id"].dropna().astype(str).unique().tolist()
                if ref_ids:
                    placeholders = ",".join([f"%s" for _ in ref_ids])
                    join_clause, ref_cols = refs_join_cols_for_source(flags.refs_source)
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            WITH x AS (
                              SELECT DISTINCT unnest(ARRAY[{placeholders}]::text[]) AS reference_id
                            )
                            SELECT {ref_cols}
                            FROM x
                            {join_clause}
                            ORDER BY 1
                        """, ref_ids)
                        rows = cur.fetchall()
                        cols = [d.name for d in cur.description]
                        df_refs_c = pd.DataFrame(rows, columns=cols)
                else:
                    df_refs_c = pd.DataFrame(columns=["reference_id","doi","title","journal","year","url"])
            else:
                df_refs_c = pd.DataFrame(columns=["reference_id","doi","title","journal","year","url"])

            # Evidence for circuit-side
            flags_now = detect_flags(conn)
            if flags_now.has_evidence and not df_conn_c.empty:
                with conn.cursor() as cur:
                    placeholders = ",".join([f"%s" for _ in circuit_ids])
                    cur.execute(f"""
                        SELECT e.evidence_id, e.circuit_id, e.receiver_id, e.reference_id,
                               e.connection_flag, e.method, e.taxon, e.modulation_type, e.output_semantics,
                               e.pointers_on_literature, e.pointers_on_figure, e.status
                        FROM evidence e
                        WHERE e.circuit_id IN ({placeholders})
                        ORDER BY e.circuit_id, e.receiver_id, e.evidence_id
                    """, circuit_ids)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_evi_c = pd.DataFrame(rows, columns=cols)
            elif not flags_now.has_evidence and not df_conn_c.empty:
                df_evi_c = df_conn_c.rename(columns={
                    "circuit_id":"circuit_id", "receiver_id":"receiver_id",
                    "method":"method", "taxon":"taxon", "reference_id":"reference_id",
                    "pointers_on_literature":"pointers_on_literature",
                    "pointers_on_figure":"pointers_on_figure"
                }).assign(connection_flag=True, evidence_id=None, modulation_type=None, output_semantics=None, status="SURROGATE")
                df_evi_c = df_evi_c[[
                    "evidence_id","circuit_id","receiver_id","reference_id","connection_flag",
                    "method","taxon","modulation_type","output_semantics","pointers_on_literature","pointers_on_figure","status"
                ]]
            else:
                df_evi_c = pd.DataFrame(columns=[
                    "evidence_id","circuit_id","receiver_id","reference_id","connection_flag",
                    "method","taxon","modulation_type","output_semantics","pointers_on_literature","pointers_on_figure","status"
                ])

            # Scores for circuit-side
            if flags_now.has_scores and flags_now.has_connections_std:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT s.* FROM scores s
                            JOIN connections_std cs ON cs.connection_id = s.connection_id
                            WHERE cs.circuit_id = ANY(%s)
                            ORDER BY s.connection_id, s.score_id
                        """, (circuit_ids,))
                        rows = cur.fetchall()
                        cols = [d.name for d in cur.description]
                        df_score_c = pd.DataFrame(rows, columns=cols)
                except Exception:
                    df_score_c = df_conn_c[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                        columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})
            else:
                if not df_conn_c.empty:
                    df_score_c = df_conn_c[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                        columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})
                else:
                    df_score_c = pd.DataFrame(columns=["circuit_id","receiver_id","reference_id","score_proxy","summary"])

            # Receiver-side connections
            if receiver_ids:
                with conn.cursor() as cur:
                    placeholders = ",".join([f"%s" for _ in receiver_ids])
                    cur.execute(f"""
                        SELECT
                          c.sender_circuit_id AS circuit_id,
                          c.receiver_circuit_id AS receiver_id,
                          c.reference_id,
                          c.measurement_method AS method,
                          c.taxon,
                          c.pointers_on_literature,
                          c.pointers_on_figure,
                          c.credibility_rating,
                          c.summarized_cr,
                          c.reviewer
                        FROM connections c
                        WHERE c.receiver_circuit_id IN ({placeholders})
                        ORDER BY c.sender_circuit_id, c.receiver_circuit_id, c.reference_id
                    """, receiver_ids)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_conn_r = pd.DataFrame(rows, columns=cols)
            else:
                df_conn_r = pd.DataFrame(columns=df_conn_c.columns)

            # References for receiver-side
            if not df_conn_r.empty:
                ref_ids_r = df_conn_r["reference_id"].dropna().astype(str).unique().tolist()
                if ref_ids_r:
                    placeholders = ",".join([f"%s" for _ in ref_ids_r])
                    join_clause, ref_cols = refs_join_cols_for_source(flags.refs_source)
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            WITH x AS (
                              SELECT DISTINCT unnest(ARRAY[{placeholders}]::text[]) AS reference_id
                            )
                            SELECT {ref_cols}
                            FROM x
                            {join_clause}
                            ORDER BY 1
                        """, ref_ids_r)
                        rows = cur.fetchall()
                        cols = [d.name for d in cur.description]
                        df_refs_r = pd.DataFrame(rows, columns=cols)
                else:
                    df_refs_r = pd.DataFrame(columns=["reference_id","doi","title","journal","year","url"])
            else:
                df_refs_r = pd.DataFrame(columns=["reference_id","doi","title","journal","year","url"])

            # Evidence for receiver-side
            if flags_now.has_evidence and not df_conn_r.empty:
                with conn.cursor() as cur:
                    placeholders = ",".join([f"%s" for _ in receiver_ids])
                    cur.execute(f"""
                        SELECT e.evidence_id, e.circuit_id, e.receiver_id, e.reference_id,
                               e.connection_flag, e.method, e.taxon, e.modulation_type, e.output_semantics,
                               e.pointers_on_literature, e.pointers_on_figure, e.status
                        FROM evidence e
                        WHERE e.receiver_id IN ({placeholders})
                        ORDER BY e.circuit_id, e.receiver_id, e.evidence_id
                    """, receiver_ids)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_evi_r = pd.DataFrame(rows, columns=cols)
            elif not flags_now.has_evidence and not df_conn_r.empty:
                df_evi_r = df_conn_r.rename(columns={
                    "circuit_id":"circuit_id", "receiver_id":"receiver_id",
                    "method":"method", "taxon":"taxon", "reference_id":"reference_id",
                    "pointers_on_literature":"pointers_on_literature",
                    "pointers_on_figure":"pointers_on_figure"
                }).assign(connection_flag=True, evidence_id=None, modulation_type=None, output_semantics=None, status="SURROGATE")
                df_evi_r = df_evi_r[[
                    "evidence_id","circuit_id","receiver_id","reference_id","connection_flag",
                    "method","taxon","modulation_type","output_semantics","pointers_on_literature","pointers_on_figure","status"
                ]]
            else:
                df_evi_r = pd.DataFrame(columns=df_evi_c.columns)

            # Scores for receiver-side
            if flags_now.has_scores and flags_now.has_connections_std:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT s.* FROM scores s
                            JOIN connections_std cs ON cs.connection_id = s.connection_id
                            WHERE cs.receiver_id = ANY(%s)
                            ORDER BY s.connection_id, s.score_id
                        """, (receiver_ids,))
                        rows = cur.fetchall()
                        cols = [d.name for d in cur.description]
                        df_score_r = pd.DataFrame(rows, columns=cols)
                except Exception:
                    df_score_r = df_conn_r[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                        columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})
            else:
                if not df_conn_r.empty:
                    df_score_r = df_conn_r[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                        columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})
                else:
                    df_score_r = pd.DataFrame(columns=["circuit_id","receiver_id","reference_id","score_proxy","summary"])

            diag = json.dumps({"flags": vars(flags_now), "circuit_matches": len(circuit_ids), "receiver_matches": len(receiver_ids)}, ensure_ascii=False, indent=2)
            return (df_circuits, df_conn_c, df_refs_c, df_evi_c, df_score_c,
                    df_receivers, df_conn_r, df_refs_r, df_evi_r, df_score_r, diag)
    except Exception as e:
        tb = traceback.format_exc()
        empty = pd.DataFrame()
        return (empty, empty, empty, empty, empty,
                empty, empty, empty, empty, empty,
                f"DB error: {e}\n\n{tb}")


# ------------------------------
# Pair lookup & Flex Pair logic
# ------------------------------

def run_pair_lookup(sender_id: str, receiver_id: str):
    sender = (sender_id or "").strip()
    receiver = (receiver_id or "").strip()
    empty = pd.DataFrame()

    if sender == "" or receiver == "":
        return gr.update(value="Sender と Receiver を入力/選択してください。"), empty, empty, empty, empty

    try:
        with psycopg2.connect(**get_dsn()) as conn:
            flags = detect_flags(conn)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM connections
                    WHERE sender_circuit_id = %s AND receiver_circuit_id = %s
                """, (sender, receiver))
                n = cur.fetchone()[0]

            if n == 0:
                return gr.update(value=f"**未存在**：Connections に `{sender}` → `{receiver}` の行はありません。"), empty, empty, empty, empty

            # 1) connections rows (pair)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                      c.sender_circuit_id AS circuit_id,
                      c.receiver_circuit_id AS receiver_id,
                      c.reference_id,
                      c.measurement_method AS method,
                      c.taxon,
                      c.pointers_on_literature,
                      c.pointers_on_figure,
                      c.credibility_rating,
                      c.summarized_cr,
                      c.reviewer
                    FROM connections c
                    WHERE c.sender_circuit_id = %s AND c.receiver_circuit_id = %s
                    ORDER BY c.reference_id
                """, (sender, receiver))
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]
                df_conn = pd.DataFrame(rows, columns=cols)

            # 2) references (distinct)
            ref_ids = df_conn["reference_id"].dropna().astype(str).unique().tolist()
            if ref_ids:
                placeholders = ",".join([f"%s" for _ in ref_ids])
                join_clause, ref_cols = refs_join_cols_for_source(flags.refs_source)
                with conn.cursor() as cur:
                    cur.execute(f"""
                        WITH x AS (
                          SELECT DISTINCT unnest(ARRAY[{placeholders}]::text[]) AS reference_id
                        )
                        SELECT {ref_cols}
                        FROM x
                        {join_clause}
                        ORDER BY 1
                    """, ref_ids)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_refs = pd.DataFrame(rows, columns=cols)
            else:
                df_refs = pd.DataFrame(columns=["reference_id","doi","title","journal","year","url"])

            # 3) evidence (pair)
            if flags.has_evidence:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT e.evidence_id, e.circuit_id, e.receiver_id, e.reference_id,
                               e.connection_flag, e.method, e.taxon, e.modulation_type, e.output_semantics,
                               e.pointers_on_literature, e.pointers_on_figure, e.status
                        FROM evidence e
                        WHERE e.circuit_id = %s AND e.receiver_id = %s
                        ORDER BY e.evidence_id
                    """, (sender, receiver))
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
                    df_evi = pd.DataFrame(rows, columns=cols)
            else:
                # fallback: connections を evidence風に
                df_evi = df_conn.rename(columns={
                    "method":"method", "taxon":"taxon", "reference_id":"reference_id",
                    "pointers_on_literature":"pointers_on_literature",
                    "pointers_on_figure":"pointers_on_figure"
                }).assign(
                    evidence_id=None, circuit_id=sender, receiver_id=receiver,
                    connection_flag=True, modulation_type=None, output_semantics=None, status="SURROGATE"
                )[["evidence_id","circuit_id","receiver_id","reference_id","connection_flag",
                   "method","taxon","modulation_type","output_semantics","pointers_on_literature","pointers_on_figure","status"]]

            # 4) scores (pair)
            if flags.has_scores and flags.has_connections_std:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT s.* FROM scores s
                            JOIN connections_std cs ON cs.connection_id = s.connection_id
                            WHERE cs.circuit_id = %s AND cs.receiver_id = %s
                            ORDER BY s.connection_id, s.score_id
                        """, (sender, receiver))
                        rows = cur.fetchall()
                        cols = [d.name for d in cur.description]
                        df_scores = pd.DataFrame(rows, columns=cols)
                except Exception:
                    df_scores = df_conn[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                        columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})
            else:
                df_scores = df_conn[["circuit_id","receiver_id","reference_id","credibility_rating","summarized_cr"]].rename(
                    columns={"credibility_rating":"score_proxy","summarized_cr":"summary"})

            msg = gr.update(value=f"**存在します**：`{sender}` → `{receiver}`（connections: {len(df_conn)} 件）")
            return msg, df_conn, df_refs, df_evi, df_scores

    except Exception as e:
        return gr.update(value=f"DB error: {e}"), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def update_counterparts(selected_region: str):
    """選ばれた領域が Sender/Receiver のどちらに現れるかを判定し、対向候補を返す"""
    rid = (selected_region or "").strip()
    if rid == "":
        return (gr.update(value="領域を選択してください。"),
                gr.update(value="Auto"),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                json.dumps({"cnt_as_sender":0,"cnt_as_receiver":0}))

    try:
        with psycopg2.connect(**get_dsn()) as conn:
            cur = conn.cursor()
            # receivers when region is sender
            cur.execute("""
                SELECT DISTINCT receiver_circuit_id
                FROM connections
                WHERE sender_circuit_id = %s
                ORDER BY 1
            """, (rid,))
            receivers = [r[0] for r in cur.fetchall() if r[0] is not None]

            # senders when region is receiver
            cur.execute("""
                SELECT DISTINCT sender_circuit_id
                FROM connections
                WHERE receiver_circuit_id = %s
                ORDER BY 1
            """, (rid,))
            senders = [r[0] for r in cur.fetchall() if r[0] is not None]

        cnt_s = len(receivers)   # region as sender → candidate receivers
        cnt_r = len(senders)     # region as receiver → candidate senders

        if cnt_s > 0 and cnt_r == 0:
            mode_val = "Use as Sender"
        elif cnt_s == 0 and cnt_r > 0:
            mode_val = "Use as Receiver"
        else:
            mode_val = "Auto"

        msg = f"選択: `{rid}` — as **Sender**: {cnt_s} 件 / as **Receiver**: {cnt_r} 件"
        return (gr.update(value=msg),
                gr.update(value=mode_val),
                gr.update(choices=receivers, value=None, interactive=(cnt_s>0 and (mode_val!='Use as Receiver'))),
                gr.update(choices=senders,   value=None, interactive=(cnt_r>0 and (mode_val!='Use as Sender'))),
                json.dumps({"cnt_as_sender":cnt_s,"cnt_as_receiver":cnt_r}))

    except Exception as e:
        return (gr.update(value=f"DB error: {e}"),
                gr.update(value="Auto"),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                json.dumps({"cnt_as_sender":0,"cnt_as_receiver":0}))


def update_counterparts_and_clear(selected_region: str):
    """候補を再計算したうえで、結果表示をクリア"""
    status, mode_update, rec_dd, snd_dd, counts = update_counterparts(selected_region)
    empty = pd.DataFrame()
    return (status, mode_update, rec_dd, snd_dd, counts, empty, empty, empty, empty)


def toggle_mode(mode: str, counts_json: str):
    """モード切替に応じて、どちらのドロップダウンを操作可能にするかを制御"""
    try:
        counts = json.loads(counts_json or "{}")
        cnt_s = int(counts.get("cnt_as_sender", 0))
        cnt_r = int(counts.get("cnt_as_receiver", 0))
    except Exception:
        cnt_s = cnt_r = 0

    if mode == "Use as Sender":
        return (gr.update(interactive=True),  gr.update(interactive=False))
    elif mode == "Use as Receiver":
        return (gr.update(interactive=False), gr.update(interactive=True))
    else:  # Auto
        if cnt_s>0 and cnt_r==0:
            return (gr.update(interactive=True),  gr.update(interactive=False))
        elif cnt_s==0 and cnt_r>0:
            return (gr.update(interactive=False), gr.update(interactive=True))
        else:
            return (gr.update(interactive=True),  gr.update(interactive(True)))


def lookup_from_flex(selected_region: str, mode: str, chosen_receiver: str, chosen_sender: str):
    """モードと選択に応じて sender/receiver を決定し、ペア検索"""
    rid = (selected_region or "").strip()
    if rid == "":
        empty = pd.DataFrame()
        return (gr.update(value="領域を先に選択してください。"), empty, empty, empty, empty)

    m = (mode or "Auto").strip()
    if m == "Use as Sender":
        sender, receiver = rid, (chosen_receiver or "").strip()
    elif m == "Use as Receiver":
        sender, receiver = (chosen_sender or "").strip(), rid
    else:
        if chosen_receiver:
            sender, receiver = rid, chosen_receiver.strip()
        elif chosen_sender:
            sender, receiver = chosen_sender.strip(), rid
        else:
            empty = pd.DataFrame()
            return (gr.update(value="対向候補を選択してください。"), empty, empty, empty, empty)

    return run_pair_lookup(sender, receiver)


def clear_results_only():
    """結果領域だけをクリア（候補リストや選択は保持）"""
    empty = pd.DataFrame()
    return (gr.update(value="表示をクリアしました。"), empty, empty, empty, empty)


def refresh_candidates_and_clear(selected_region: str):
    """現在の領域から候補を再計算し、結果をクリア"""
    return update_counterparts_and_clear(selected_region)


# ------------------------------
# UI
# ------------------------------

def build_ui():
    with gr.Blocks(title="WholeBIF-RDB – Query & Pair Tools") as demo:
        gr.Markdown("# WholeBIF-RDB – Query & Pair Tools")

        # =========================
        # Tab 1: Query Explorer
        # =========================
        with gr.Tab("Query Explorer"):
            gr.Markdown("クエリに「類似」する Circuit / Receiver を探索し、関連する Connections / References / Evidence / Scores を**別エリア**に表示。")
            with gr.Row():
                query = gr.Textbox(label="キーワード（例: 'Thalamus', 'CA1' など）", placeholder="入力すると候補が出ます", scale=4)
                limit = gr.Slider(label="最大件数", minimum=5, maximum=100, value=20, step=5, scale=1)
                btn = gr.Button("Search", variant="primary", scale=1)

            suggest = gr.Dropdown(label="候補 Circuit ID（クリックで確定）", choices=[], interactive=True, allow_custom_value=False)

            diag = gr.Code(label="診断情報（flags / ヒット件数）", interactive=False)

            with gr.Accordion("A. Circuit に類似（Circuit ID / Names）", open=True):
                gr.Markdown("**(1) 類似 Circuit**")
                df_circuits = gr.Dataframe(label="Matched Circuits", interactive=False, wrap=True)

                gr.Markdown("**(2) 関係を持つ Connections（送信: Matched Circuits → 受信）**")
                df_conn_c = gr.Dataframe(label="Connections from Circuit matches", interactive=False, wrap=True)

                gr.Markdown("**(3) 関係する References**（Connections の Reference ID に基づく）")
                df_refs_c = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)

                gr.Markdown("**(4) 関係する Evidence**（Evidence が無ければ、Connections を代替表示）")
                df_evi_c = gr.Dataframe(label="Evidence (outgoing)", interactive=False, wrap=True)

                gr.Markdown("**(5) 関係する Scores**（scores が無い場合は connections の rating を代替表示）")
                df_score_c = gr.Dataframe(label="Scores (or proxy)", interactive=False, wrap=True)

            with gr.Accordion("B. Receiver に類似（Connections.receiver_circuit_id）", open=True):
                gr.Markdown("**(1) 類似 Receiver ID**")
                df_receivers = gr.Dataframe(label="Matched Receivers", interactive=False, wrap=True)

                gr.Markdown("**(2) 関係を持つ Connections（送信 → 受信: Matched Receivers）**")
                df_conn_r = gr.Dataframe(label="Connections to Receiver matches", interactive=False, wrap=True)

                gr.Markdown("**(3) 関係する References**")
                df_refs_r = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)

                gr.Markdown("**(4) 関係する Evidence**（Evidence が無ければ、Connections を代替表示）")
                df_evi_r = gr.Dataframe(label="Evidence (incoming)", interactive=False, wrap=True)

                gr.Markdown("**(5) 関係する Scores**（scores が無い場合は connections の rating を代替表示）")
                df_score_r = gr.Dataframe(label="Scores (or proxy)", interactive=False, wrap=True)

            btn.click(
                fn=run_query,
                inputs=[query, limit],
                outputs=[df_circuits, df_conn_c, df_refs_c, df_evi_c, df_score_c,
                         df_receivers, df_conn_r, df_refs_r, df_evi_r, df_score_r, diag]
            )

            query.input(fn=suggest_circuit_ids, inputs=[query], outputs=[suggest], queue=False)
            evt = suggest.select(fn=apply_selection_to_text, inputs=[suggest], outputs=[query])
            evt.then(
                fn=run_query,
                inputs=[query, limit],
                outputs=[df_circuits, df_conn_c, df_refs_c, df_evi_c, df_score_c,
                         df_receivers, df_conn_r, df_refs_r, df_evi_r, df_score_r, diag]
            )

        # =========================
        # Tab 2: Pair Lookup
        # =========================
        with gr.Tab("Pair Lookup"):
            gr.Markdown("**Sender（Circuit ID）** と **Receiver（Receiver ID）** を指定して、Connections の**存在確認**と、関連する **References / Evidence / Scores** を別エリアで表示します。")

            with gr.Row():
                sender_q = gr.Textbox(label="Sender (Circuit ID)", placeholder="例: CA1 など、候補から選ぶのがおすすめ", scale=3)
                receiver_q = gr.Textbox(label="Receiver (Receiver ID)", placeholder="例: DG など、候補から選ぶのがおすすめ", scale=3)
                btn_pair = gr.Button("Lookup", variant="primary")

            with gr.Row():
                suggest_sender = gr.Dropdown(label="候補（Circuit ID）", choices=[], interactive=True, allow_custom_value=False)
                suggest_receiver = gr.Dropdown(label="候補（Receiver ID）", choices=[], interactive=True, allow_custom_value=False)

            status = gr.Markdown()

            with gr.Row():
                df_pair_conn = gr.Dataframe(label="Connections (Sender → Receiver)", interactive=False, wrap=True)

            with gr.Row():
                df_pair_refs = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)

            with gr.Row():
                df_pair_evi = gr.Dataframe(label="Evidence (pair)", interactive=False, wrap=True)

            with gr.Row():
                df_pair_scores = gr.Dataframe(label="Scores (pair | proxy)", interactive=False, wrap=True)

            # actions
            btn_pair.click(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                           outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])

            # suggestions (live)
            sender_q.input(fn=suggest_circuit_ids, inputs=[sender_q], outputs=[suggest_sender], queue=False)
            receiver_q.input(fn=suggest_receiver_ids, inputs=[receiver_q], outputs=[suggest_receiver], queue=False)

            # choose suggestion -> reflect to textbox -> auto lookup
            evt_s = suggest_sender.select(fn=apply_selection_to_text, inputs=[suggest_sender], outputs=[sender_q])
            evt_r = suggest_receiver.select(fn=apply_selection_to_text, inputs=[suggest_receiver], outputs=[receiver_q])
            evt_s.then(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                       outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])
            evt_r.then(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                       outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])

        # =========================
        # Tab 3: Flex Pair Finder
        # =========================
        with gr.Tab("Flex Pair Finder"):
            gr.Markdown("1つの脳領域から出発し、Sender/Receiver のどちらでも柔軟に対向候補を絞り込み、ペアを確定します。")

            with gr.Row():
                region_text = gr.Textbox(label="領域（ID/Names/Receiverに部分一致）", placeholder="2文字以上でサジェスト", scale=3)
                region_suggest = gr.Dropdown(label="候補（クリックで確定）", choices=[], interactive=True, allow_custom_value=False, scale=2)
                mode = gr.Radio(label="方向", choices=["Auto","Use as Sender","Use as Receiver"], value="Auto", scale=2)

            with gr.Row():
                receivers_dd = gr.Dropdown(label="対向候補（あなたが Sender のときの Receivers）", choices=[], interactive=False, allow_custom_value=False, scale=3)
                senders_dd   = gr.Dropdown(label="対向候補（あなたが Receiver のときの Senders）", choices=[], interactive=False, allow_custom_value=False, scale=3)

            with gr.Row():
                btn_lookup   = gr.Button("Lookup Pair", variant="primary")
                btn_clear    = gr.Button("Clear Results")
                btn_refresh  = gr.Button("Refresh Candidates")

            status2 = gr.Markdown()

            df_pair_conn2   = gr.Dataframe(label="Connections (pair)", interactive=False, wrap=True)
            df_pair_refs2   = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)
            df_pair_evi2    = gr.Dataframe(label="Evidence (pair)", interactive=False, wrap=True)
            df_pair_scores2 = gr.Dataframe(label="Scores (pair | proxy)", interactive=False, wrap=True)

            counts_state = gr.State(value=json.dumps({"cnt_as_sender":0,"cnt_as_receiver":0}))

            region_text.input(fn=suggest_any_region, inputs=[region_text], outputs=[region_suggest], queue=False)

            evt = region_suggest.select(fn=apply_selection_to_text, inputs=[region_suggest], outputs=[region_text])
            evt.then(fn=update_counterparts_and_clear, inputs=[region_text],
                     outputs=[status2, mode, receivers_dd, senders_dd, counts_state, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])

            mode.change(fn=toggle_mode, inputs=[mode, counts_state],
                        outputs=[receivers_dd, senders_dd])

            btn_lookup.click(fn=lookup_from_flex, inputs=[region_text, mode, receivers_dd, senders_dd],
                             outputs=[status2, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])

            btn_clear.click(fn=clear_results_only, inputs=[], outputs=[status2, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])
            btn_refresh.click(fn=refresh_candidates_and_clear, inputs=[region_text],
                              outputs=[status2, mode, receivers_dd, senders_dd, counts_state, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])

            receivers_dd.select(fn=lookup_from_flex, inputs=[region_text, mode, receivers_dd, senders_dd],
                                outputs=[status2, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])
            senders_dd.select(fn=lookup_from_flex, inputs=[region_text, mode, receivers_dd, senders_dd],
                              outputs=[status2, df_pair_conn2, df_pair_refs2, df_pair_evi2, df_pair_scores2])

        gr.Markdown("— ヒント: `CREATE EXTENSION IF NOT EXISTS pg_trgm;` を有効にするとサジェスト/検索の精度が上がります。—")

    return demo


# ------------------------------
# Public launcher (CLI)
# ------------------------------

def _parse_auth(auth_str: str):
    if not auth_str:
        return None
    items = [p.strip() for p in auth_str.split(",") if p.strip()]
    creds: List[Tuple[str, str]] = []
    for it in items:
        if ":" not in it:
            continue
        u, p = it.split(":", 1)
        creds.append((u, p))
    if not creds:
        return None
    if len(creds) == 1:
        return creds[0]
    return creds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (ephemeral)")
    parser.add_argument("--host", default=os.getenv("GRADIO_HOST", "127.0.0.1"), help="Bind address (e.g., 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT", os.getenv("PORT", "7860"))), help="Port")
    parser.add_argument("--auth", default=os.getenv("GRADIO_AUTH", ""), help="Basic auth (user:pass or 'u1:p1,u2:p2')")
    args = parser.parse_args()

    # env override for share
    share_env = os.getenv("GRADIO_SHARE", "")
    share_flag = args.share or share_env.lower() in ("1", "true", "yes", "y")

    auth = _parse_auth(args.auth)

    load_dotenv()
    app = build_ui()
    app.queue()
    app.launch(
        share=share_flag,
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        show_api=False,
    )


if __name__ == "__main__":
    main()
