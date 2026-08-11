"""`hcw search` — find the moment, across everything you ever recorded.

Three signals, none of which gets the last word:

1. Vector recall over every passage. Finds "what did we say about pricing" in a
   conversation that only ever said "the number we charge".
2. FTS5 recall in parallel. Catches the exact product name the embedding blurred.
3. A rerank pass over the union, which reads query and passage together instead
   of comparing two vectors made in isolation.

They are combined with reciprocal rank fusion rather than by letting the last
one overwrite the others. That is not academic caution: on the first real
transcript indexed with this tool, vector recall put the right passage first by
a wide margin (0.44 against 0.25) and the reranker flipped the pair on scores
that were nearly tied (0.127 against 0.104). A reranker is good at separating
candidates that are genuinely close and unreliable when they are not, so it
votes here instead of deciding.

Without an API key or an embedding endpoint the first and third signals fall
away, the search degrades to keyword matching, and it says so rather than
quietly returning worse answers.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich.text import Text

from ..api import ApiError, HelmcodeClient
from ..config import Config
from ..ui import console, hairline
from .index import connect, load_vectors

VECTOR_RECALL = 50
KEYWORD_RECALL = 20

# Reciprocal rank fusion's damping constant. 60 is the value from the original
# paper and the one every search stack has copied since; it makes the difference
# between rank 1 and rank 2 meaningful and the difference between rank 40 and
# rank 41 negligible.
RRF_K = 60


@dataclass
class Hit:
    passage_id: int
    meeting_title: str
    meeting_date: str
    start: float
    speaker: str
    text: str
    score: float
    source: str


def run_search(config: Config, query: str, *, limit: int = 5) -> None:
    """The CLI entry point: search, then print."""
    hits, semantic = search_hits(config, query, limit=limit)
    if not hits:
        console().print(f"\n  Nothing found for “{query}”.\n", style="tertiary")
        return
    _print_hits(query, hits, semantic=semantic)


def search_hits(config: Config, query: str, *, limit: int = 5) -> tuple[list[Hit], bool]:
    """Search, and return the hits plus whether semantic recall was available.

    Kept separate from `run_search` so anything other than a terminal — a local
    UI, a script, an editor plugin — can use the search without inheriting a
    print statement.
    """
    if not config.db_path.is_file():
        raise RuntimeError(
            f"No search index at {config.db_path}. Process a meeting first with `hcw process`."
        )

    connection = connect(config.db_path)
    client: HelmcodeClient | None = None
    try:
        semantic = True
        vector_hits: list[Hit] = []

        if config.api_key:
            try:
                client = HelmcodeClient(config)
                vector_hits = _vector_search(connection, client, query, config.embed_model)
            except ApiError:
                semantic = False
        else:
            semantic = False

        keyword_hits = _keyword_search(connection, query)
        candidates = _merge_hits(vector_hits, keyword_hits)

        if not candidates:
            return [], semantic

        rankings = [
            [hit.passage_id for hit in sorted(vector_hits, key=lambda h: h.score, reverse=True)],
            [hit.passage_id for hit in sorted(keyword_hits, key=lambda h: h.score, reverse=True)],
        ]

        # Rerank whenever there is anything to order. An earlier version skipped
        # this when the candidate set fitted inside `limit`, on the theory that
        # showing every candidate makes ordering moot. It does not: the order is
        # the answer, and that shortcut made `-n 8` and `-n 2` disagree about
        # which passage came first.
        if client is not None and len(candidates) > 1:
            try:
                rankings.append(_rerank_ranking(client, config.rerank_model, query, candidates))
            except ApiError:
                pass

        return _fuse(candidates, rankings)[:limit], semantic
    finally:
        connection.close()
        if client is not None:
            client.close()


def _fuse(candidates: list[Hit], rankings: list[list[int]]) -> list[Hit]:
    """Reciprocal rank fusion: every signal votes, none of them decides alone."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, passage_id in enumerate(ranking):
            fused[passage_id] += 1.0 / (RRF_K + rank + 1)

    for hit in candidates:
        hit.score = fused[hit.passage_id]
    return sorted(candidates, key=lambda hit: hit.score, reverse=True)


def _vector_search(
    connection: sqlite3.Connection, client: HelmcodeClient, query: str, model: str
) -> list[Hit]:
    ids, matrix = load_vectors(connection)
    if not ids:
        return []

    vector = np.asarray(client.embed([query], model=model)[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm

    scores = matrix @ vector
    top = np.argsort(scores)[::-1][:VECTOR_RECALL]
    return _hydrate(connection, [(ids[i], float(scores[i]), "vector") for i in top])


def _keyword_search(connection: sqlite3.Connection, query: str) -> list[Hit]:
    # FTS5 treats punctuation as syntax; a natural-language question would be a
    # parse error. Feed it the words, OR-ed.
    terms = [word for word in "".join(c if c.isalnum() else " " for c in query).split() if word]
    if not terms:
        return []
    match = " OR ".join(terms)
    try:
        rows = connection.execute(
            "SELECT rowid, rank FROM passages_fts WHERE passages_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, KEYWORD_RECALL),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    # bm25 rank is negative and unbounded; only the ordering matters here,
    # because the rerank pass decides the final order anyway.
    return _hydrate(connection, [(row["rowid"], -float(row["rank"]), "keyword") for row in rows])


def _hydrate(
    connection: sqlite3.Connection, scored: list[tuple[int, float, str]]
) -> list[Hit]:
    if not scored:
        return []
    by_id = {passage_id: (score, source) for passage_id, score, source in scored}
    placeholders = ",".join("?" * len(by_id))
    rows = connection.execute(
        f"SELECT p.id, p.start, p.speaker, p.text, m.title, m.date "
        f"FROM passages p JOIN meetings m ON m.id = p.meeting_id "
        f"WHERE p.id IN ({placeholders})",
        tuple(by_id),
    ).fetchall()

    hits = []
    for row in rows:
        score, source = by_id[row["id"]]
        hits.append(
            Hit(
                passage_id=row["id"],
                meeting_title=row["title"],
                meeting_date=row["date"],
                start=row["start"],
                speaker=row["speaker"],
                text=row["text"],
                score=score,
                source=source,
            )
        )
    return hits


def _merge_hits(primary: list[Hit], secondary: list[Hit]) -> list[Hit]:
    merged = {hit.passage_id: hit for hit in primary}
    for hit in secondary:
        if hit.passage_id in merged:
            merged[hit.passage_id].source = "both"
        else:
            merged[hit.passage_id] = hit
    return list(merged.values())


def _rerank_ranking(
    client: HelmcodeClient, model: str, query: str, hits: list[Hit]
) -> list[int]:
    """The reranker's opinion as an ordering of passage ids, not as a verdict.

    No `top_n`: the fusion wants the whole ordering, and a truncated list would
    silently give every dropped passage the same rank.
    """
    results = client.rerank(query, [hit.text for hit in hits], model=model)
    ranking: list[int] = []
    for result in results:
        index = result.get("index")
        if index is not None and 0 <= index < len(hits):
            ranking.append(hits[index].passage_id)
    return ranking


def _print_hits(query: str, hits: list[Hit], *, semantic: bool) -> None:
    console().print()
    # The eyebrow style letter-spaces its text, which reads as a label and as
    # nonsense on anything longer. The query goes below it, unspaced.
    hairline("results")
    console().print(Text(f"  {query}", style="primary"))
    if not semantic:
        console().print(
            Text(
                "  keyword search only — no embeddings available, so meaning-based "
                "matches are missing",
                style="warn",
            )
        )
    console().print()

    for hit in hits:
        stamp = f"{int(hit.start) // 60:02d}:{int(hit.start) % 60:02d}"
        header = Text.assemble(
            ("  ", ""),
            (hit.meeting_title, "primary"),
            ("  ", ""),
            (hit.meeting_date, "tertiary"),
            ("  ", ""),
            (stamp, "timestamp"),
            ("  ", ""),
            (hit.speaker, "accent"),
        )
        console().print(header)
        console().print(Text(f"  {hit.text}", style="secondary"))
        console().print()
