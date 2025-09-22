#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradio_wholebif_query_app_flexpair_public_v2.py
-----------------------------------------------
Fixes for share=True case where live suggestions may not appear:
- Uses a psycopg2 connection pool (reduces connect storms)
- Suggestion functions are bound to both `.input` and `.change`
- Adds explicit "Suggest" buttons to manually refresh candidates
- Gradio queue enabled with small concurrency for stability

Usage:
  python gradio_wholebif_query_app_flexpair_public_v2.py --share --auth user:pass
  python gradio_wholebif_query_app_flexpair_public_v2.py --host 0.0.0.0 --port 7860 --auth user:pass
"""

import os
import json
import argparse
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import gradio as gr


# ------------------------------
# DB & pooling
# ------------------------------

@dataclass
class DBFlags:
    has_evidence: bool
    has_refs_view: bool
    refs_source: str
    has_scores: bool
    has_pg_trgm: bool
    has_connections_std: bool


_POOL: SimpleConnectionPool | None = None

def get_dsn() -> Dict[str, Any]:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db   = os.getenv("POSTGRES_DB", "wholebif_rdb")
    user = os.getenv("POSTGRES_USER", "wholebif")
    pwd  = os.getenv("POSTGRES_PASSWORD", "")
    return dict(host=host, port=port, dbname=db, user=user, password=pwd)

def pool() -> SimpleConnectionPool:
    global _POOL
    if _POOL is None:
        dsn = get_dsn()
        minc = int(os.getenv("DB_POOL_MIN", "1"))
        maxc = int(os.getenv("DB_POOL_MAX", "6"))
        _POOL = SimpleConnectionPool(minc, maxc, **dsn)
    return _POOL

def with_conn(fn):
    """Decorator to borrow/return a pooled connection"""
    def wrapper(*args, **kwargs):
        p = pool()
        conn = p.getconn()
        try:
            conn.autocommit = True
            return fn(conn, *args, **kwargs)
        finally:
            p.putconn(conn)
    return wrapper


# ------------------------------
# Flags / detection
# ------------------------------

@with_conn
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
# Suggestors (pooled)
# ------------------------------

@with_conn
def _suggest_circuit_ids(conn, q: str, topn: int = 12) -> List[str]:
    flags = detect_flags()
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
        return [r[0] for r in cur.fetchall()]

def suggest_circuit_ids(partial: str, topn: int = 12):
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        return gr.update(choices=_suggest_circuit_ids(q, topn), value=None)
    except Exception as e:
        print("suggest_circuit_ids error:", e)
        return gr.update(choices=[], value=None)


@with_conn
def _suggest_receiver_ids(conn, q: str, topn: int = 12) -> List[str]:
    flags = detect_flags()
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
        return [r[0] for r in cur.fetchall()]

def suggest_receiver_ids(partial: str, topn: int = 12):
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        return gr.update(choices=_suggest_receiver_ids(q, topn), value=None)
    except Exception as e:
        print("suggest_receiver_ids error:", e)
        return gr.update(choices=[], value=None)


@with_conn
def _suggest_any_region(conn, q: str, topn: int = 12) -> List[str]:
    flags = detect_flags()
    cur = conn.cursor()
    items: List[Tuple[str, float]] = []  # (id, score)
    if flags.has_pg_trgm:
        cur.execute("""
            SELECT circuit_id,
                   GREATEST(similarity(circuit_id, %(q)s), similarity(COALESCE(names,''), %(q)s)) AS sim
            FROM circuits
            WHERE circuit_id %% %(q)s OR COALESCE(names,'') %% %(q)s
            ORDER BY sim DESC LIMIT %(limit)s
        """, {"q": q, "limit": topn})
        items += [(r[0], float(r[1])) for r in cur.fetchall()]

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

    best: Dict[str, float] = {}
    for id_, s in items:
        if id_ is None: continue
        if id_ not in best or s > best[id_]:
            best[id_] = s
    choices = [k for k,_ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:topn]
    return choices

def suggest_any_region(partial: str, topn: int = 12):
    q = (partial or "").strip()
    if len(q) < 2:
        return gr.update(choices=[], value=None)
    try:
        return gr.update(choices=_suggest_any_region(q, topn), value=None)
    except Exception as e:
        print("suggest_any_region error:", e)
        return gr.update(choices=[], value=None)


def apply_selection_to_text(selected: str):
    return gr.update(value=(selected or ""))


# ------------------------------
# Pair lookup core (pooled)
# ------------------------------

@with_conn
def _pair_exists(conn, sender: str, receiver: str) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM connections
            WHERE sender_circuit_id = %s AND receiver_circuit_id = %s
        """, (sender, receiver))
        return cur.fetchone()[0]

@with_conn
def _fetch_pair_details(conn, sender: str, receiver: str, flags: DBFlags):
    # connections
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

    # references
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

    # evidence
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
        df_evi = df_conn.rename(columns={
            "method":"method", "taxon":"taxon", "reference_id":"reference_id",
            "pointers_on_literature":"pointers_on_literature",
            "pointers_on_figure":"pointers_on_figure"
        }).assign(
            evidence_id=None, circuit_id=sender, receiver_id=receiver,
            connection_flag=True, modulation_type=None, output_semantics=None, status="SURROGATE"
        )[["evidence_id","circuit_id","receiver_id","reference_id","connection_flag",
           "method","taxon","modulation_type","output_semantics","pointers_on_literature","pointers_on_figure","status"]]

    # scores
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

    return df_conn, df_refs, df_evi, df_scores


def run_pair_lookup(sender_id: str, receiver_id: str):
    sender = (sender_id or "").strip()
    receiver = (receiver_id or "").strip()
    empty = pd.DataFrame()

    if sender == "" or receiver == "":
        return gr.update(value="Sender と Receiver を入力/選択してください。"), empty, empty, empty, empty

    try:
        flags = detect_flags()
        n = _pair_exists(sender, receiver)
        if n == 0:
            return gr.update(value=f"**未存在**：Connections に `{sender}` → `{receiver}` の行はありません。"), empty, empty, empty, empty

        df_conn, df_refs, df_evi, df_scores = _fetch_pair_details(sender, receiver, flags)
        msg = gr.update(value=f"**存在します**：`{sender}` → `{receiver}`（connections: {len(df_conn)} 件）")
        return msg, df_conn, df_refs, df_evi, df_scores
    except Exception as e:
        print("run_pair_lookup error:", e)
        return gr.update(value=f"DB error: {e}"), empty, empty, empty, empty


# ------------------------------
# Flex Pair logic (pooled)
# ------------------------------

@with_conn
def _counterparts(conn, rid: str) -> Tuple[List[str], List[str]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT receiver_circuit_id
            FROM connections
            WHERE sender_circuit_id = %s
            ORDER BY 1
        """, (rid,))
        receivers = [r[0] for r in cur.fetchall() if r[0] is not None]

        cur.execute("""
            SELECT DISTINCT sender_circuit_id
            FROM connections
            WHERE receiver_circuit_id = %s
            ORDER BY 1
        """, (rid,))
        senders = [r[0] for r in cur.fetchall() if r[0] is not None]

    return receivers, senders


def update_counterparts(selected_region: str):
    rid = (selected_region or "").strip()
    if rid == "":
        return (gr.update(value="領域を選択してください。"),
                gr.update(value="Auto"),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                json.dumps({"cnt_as_sender":0,"cnt_as_receiver":0}))
    try:
        receivers, senders = _counterparts(rid)
        cnt_s, cnt_r = len(receivers), len(senders)
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
        print("update_counterparts error:", e)
        return (gr.update(value=f"DB error: {e}"),
                gr.update(value="Auto"),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                json.dumps({"cnt_as_sender":0,"cnt_as_receiver":0}))


def update_counterparts_and_clear(selected_region: str):
    status, mode_update, rec_dd, snd_dd, counts = update_counterparts(selected_region)
    empty = pd.DataFrame()
    return (status, mode_update, rec_dd, snd_dd, counts, empty, empty, empty, empty)


def toggle_mode(mode: str, counts_json: str):
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
    else:
        if cnt_s>0 and cnt_r==0:
            return (gr.update(interactive=True),  gr.update(interactive(False)))
        elif cnt_s==0 and cnt_r>0:
            return (gr.update(interactive(False)), gr.update(interactive=True))
        else:
            return (gr.update(interactive=True),  gr.update(interactive=True))


def lookup_from_flex(selected_region: str, mode: str, chosen_receiver: str, chosen_sender: str):
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
    empty = pd.DataFrame()
    return (gr.update(value="表示をクリアしました。"), empty, empty, empty, empty)


def refresh_candidates_and_clear(selected_region: str):
    return update_counterparts_and_clear(selected_region)


# ------------------------------
# UI
# ------------------------------

def build_ui():
    with gr.Blocks(title="WholeBIF-RDB – Query & Pair Tools (public v2)") as demo:
        gr.Markdown("# WholeBIF-RDB – Query & Pair Tools (public v2)")

        # ===== Query Explorer =====
        with gr.Tab("Query Explorer"):
            with gr.Row():
                query = gr.Textbox(label="キーワード", placeholder="2文字以上でサジェスト", scale=4)
                limit = gr.Slider(label="最大件数", minimum=5, maximum=100, value=20, step=5, scale=1)
                btn = gr.Button("Search", variant="primary", scale=1)
                btn_suggest = gr.Button("Suggest", scale=1)

            suggest = gr.Dropdown(label="候補 Circuit ID", choices=[], interactive=True, allow_custom_value=False)

            diag = gr.Code(label="診断情報", interactive=False)

            with gr.Accordion("A. Circuit に類似", open=True):
                df_circuits = gr.Dataframe(label="Matched Circuits", interactive=False, wrap=True)
                df_conn_c = gr.Dataframe(label="Connections from Circuit matches", interactive=False, wrap=True)
                df_refs_c = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)
                df_evi_c = gr.Dataframe(label="Evidence (outgoing)", interactive=False, wrap=True)
                df_score_c = gr.Dataframe(label="Scores (or proxy)", interactive=False, wrap=True)

            with gr.Accordion("B. Receiver に類似", open=True):
                df_receivers = gr.Dataframe(label="Matched Receivers", interactive=False, wrap=True)
                df_conn_r = gr.Dataframe(label="Connections to Receiver matches", interactive=False, wrap=True)
                df_refs_r = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)
                df_evi_r = gr.Dataframe(label="Evidence (incoming)", interactive=False, wrap=True)
                df_score_r = gr.Dataframe(label="Scores (or proxy)", interactive=False, wrap=True)

            btn.click(
                fn=run_query,
                inputs=[query, limit],
                outputs=[df_circuits, df_conn_c, df_refs_c, df_evi_c, df_score_c,
                         df_receivers, df_conn_r, df_refs_r, df_evi_r, df_score_r, diag]
            )

            # live suggest on input + change + manual button
            query.input(fn=suggest_circuit_ids, inputs=[query], outputs=[suggest], queue=False)
            query.change(fn=suggest_circuit_ids, inputs=[query], outputs=[suggest])
            btn_suggest.click(fn=suggest_circuit_ids, inputs=[query], outputs=[suggest])

            evt = suggest.select(fn=apply_selection_to_text, inputs=[suggest], outputs=[query])
            evt.then(fn=run_query, inputs=[query, limit],
                     outputs=[df_circuits, df_conn_c, df_refs_c, df_evi_c, df_score_c,
                              df_receivers, df_conn_r, df_refs_r, df_evi_r, df_score_r, diag])

        # ===== Pair Lookup =====
        with gr.Tab("Pair Lookup"):
            with gr.Row():
                sender_q = gr.Textbox(label="Sender (Circuit ID)", placeholder="2文字以上でサジェスト", scale=3)
                receiver_q = gr.Textbox(label="Receiver (Receiver ID)", placeholder="2文字以上でサジェスト", scale=3)
                btn_pair = gr.Button("Lookup", variant="primary")
                btn_suggest_s = gr.Button("Suggest Sender")
                btn_suggest_r = gr.Button("Suggest Receiver")

            suggest_sender = gr.Dropdown(label="候補（Circuit ID）", choices=[], interactive=True, allow_custom_value=False)
            suggest_receiver = gr.Dropdown(label="候補（Receiver ID）", choices=[], interactive=True, allow_custom_value=False)

            status = gr.Markdown()
            df_pair_conn = gr.Dataframe(label="Connections (Sender → Receiver)", interactive=False, wrap=True)
            df_pair_refs = gr.Dataframe(label="References (distinct)", interactive=False, wrap=True)
            df_pair_evi = gr.Dataframe(label="Evidence (pair)", interactive=False, wrap=True)
            df_pair_scores = gr.Dataframe(label="Scores (pair | proxy)", interactive=False, wrap=True)

            btn_pair.click(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                           outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])

            sender_q.input(fn=suggest_circuit_ids, inputs=[sender_q], outputs=[suggest_sender], queue=False)
            sender_q.change(fn=suggest_circuit_ids, inputs=[sender_q], outputs=[suggest_sender])
            btn_suggest_s.click(fn=suggest_circuit_ids, inputs=[sender_q], outputs=[suggest_sender])

            receiver_q.input(fn=suggest_receiver_ids, inputs=[receiver_q], outputs=[suggest_receiver], queue=False)
            receiver_q.change(fn=suggest_receiver_ids, inputs=[receiver_q], outputs=[suggest_receiver])
            btn_suggest_r.click(fn=suggest_receiver_ids, inputs=[receiver_q], outputs=[suggest_receiver])

            evt_s = suggest_sender.select(fn=apply_selection_to_text, inputs=[suggest_sender], outputs=[sender_q])
            evt_r = suggest_receiver.select(fn=apply_selection_to_text, inputs=[suggest_receiver], outputs=[receiver_q])
            evt_s.then(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                       outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])
            evt_r.then(fn=run_pair_lookup, inputs=[sender_q, receiver_q],
                       outputs=[status, df_pair_conn, df_pair_refs, df_pair_evi, df_pair_scores])

        # ===== Flex Pair Finder =====
        with gr.Tab("Flex Pair Finder"):
            with gr.Row():
                region_text = gr.Textbox(label="領域（ID/Names/Receiver）", placeholder="2文字以上でサジェスト", scale=3)
                region_suggest = gr.Dropdown(label="候補（クリックで確定）", choices=[], interactive=True, allow_custom_value=False, scale=2)
                mode = gr.Radio(label="方向", choices=["Auto","Use as Sender","Use as Receiver"], value="Auto", scale=2)
                btn_suggest_region = gr.Button("Suggest", scale=1)

            receivers_dd = gr.Dropdown(label="対向候補（あなたが Sender のときの Receivers）", choices=[], interactive=False, allow_custom_value=False)
            senders_dd   = gr.Dropdown(label="対向候補（あなたが Receiver のときの Senders）", choices=[], interactive=False, allow_custom_value=False)

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

            # live + change + manual suggest
            region_text.input(fn=suggest_any_region, inputs=[region_text], outputs=[region_suggest], queue=False)
            region_text.change(fn=suggest_any_region, inputs=[region_text], outputs=[region_suggest])
            btn_suggest_region.click(fn=suggest_any_region, inputs=[region_text], outputs=[region_suggest])

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
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("GRADIO_CONCURRENCY","2")), help="Queue concurrency")
    args = parser.parse_args()

    share_env = os.getenv("GRADIO_SHARE", "")
    share_flag = args.share or share_env.lower() in ("1", "true", "yes", "y")

    auth = _parse_auth(args.auth)

    load_dotenv()
    app = build_ui()
    app.queue(concurrency_count=args.concurrency, max_size=64)
    app.launch(
        share=share_flag,
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        show_api=False,
    )


if __name__ == "__main__":
    main()
