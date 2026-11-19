"""Hybrid retrieval: keyword search and vector search over the run's documents, merged.

CONCEPT: self-rag, choice A (D-52). Concept 2's live check asked the stored
annual report "How much of Symbotic's revenue comes from Walmart?": vector
search returned three chunks that do not mention Walmart, while 94 chunks on
30 pages do. An embedding captures the general meaning of a text, so a
question that hinges on one name is matched to chunks that sound like the
report in general. Keyword search finds the name exactly. This module runs
both and merges their rankings:

    question -> vector search (FAISS, the store)  -> best 20 chunk ids, ranked
             -> keyword search (BM25, built here) -> best 20 chunk ids, ranked
             -> Reciprocal Rank Fusion            -> the best 6 chunks

Both searches open only the documents asked for, so a run can never retrieve
another run's chunks (TC-RES-10): the rule holds by what is loaded, for the
keyword index as for the vector index.

CONCEPT: self-rag, D-53. Measured on the annual report, a question as the
planner writes it ("How much of Symbotic's revenue comes from Walmart?")
puts the answer chunk 10th by meaning and 18th by keyword; reworded as
search words ("Walmart share of total revenue largest customer") it is 1st
in both. So D-53 had the Self-RAG subgraph reword every sub-question into a
search query (P-15 v1) first; D-56 below replaced that query with a passage.
This module searches whatever texts it is given.

CONCEPT: self-rag, D-56 (multi-query). On Voxtral Small, P-15's search
queries missed the answer chunk (below rank 20), while a passage written the
way the report would state the answer (HyDE) found it 1st. The subgraph now
searches with two texts at once, the question as asked and that passage, by
retrieve_for_texts(): each text is searched both ways and merged as above,
then the per-text rankings are merged the same way. All texts go in one
embedding request, so a second text costs no extra request. Measured: the
question alone ranks the answer 9th, the passage alone 1st, both merged 3rd;
the question is kept so a misleading passage cannot steer the search alone.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from graph.document_store import DocumentStore, RetrievedChunk, StoredChunk
from graph.llm import ModelGateway

logger = logging.getLogger(__name__)

# Chunks handed to the grader (self-rag, choice D).
RETRIEVED_CHUNKS = 6
# How many chunks each search contributes before the merge: enough that a
# chunk ranked low by one search but high by the other still takes part.
CANDIDATES_PER_SEARCH = 20
# The constant of Reciprocal Rank Fusion, from the method's original paper
# (Cormack, Clarke and Buettcher) and the usual default: it keeps one list's
# first place from outweighing a chunk that both lists rank well.
RANK_FUSION_CONSTANT = 60
WORD_PATTERN = re.compile(r"\w+")
# English function words, left out of keyword search on both sides. Keyword
# search is here for what vector search misses (names, numbers, terms);
# these words never carry that. Found while building: with them kept, the
# question "What does the company automate?" made "The board meets four
# times a year." a keyword match through "the", and the merge then ranked
# that chunk, found by both searches, above the true answer found by meaning
# alone. In a long document BM25 gives such words little weight, but in a
# short one it cannot: they are in every chunk, and so is the match.
STOP_WORDS = frozenset(
    "a an the of to in on at by for from with as and or but not no is are was were be been "
    "being it its this that these those what which who whom whose how when where why does "
    "do did done has have had much many s".split()
)


def keyword_tokens(text: str) -> list[str]:
    """The words of a text for keyword search: lower case, letters and digits, no function words.

    "Symbotic's revenue (fiscal 2025)" gives symbotic, revenue, fiscal, 2025
    (the possessive "s" and other function words are dropped, see STOP_WORDS).
    No stemming: names and numbers, the words vector search misses, are
    matched exactly.
    """
    return [word for word in WORD_PATTERN.findall(text.lower()) if word not in STOP_WORDS]


def reciprocal_rank_fusion(
    ranked_chunk_id_lists: list[list[str]],
    rank_constant: int = RANK_FUSION_CONSTANT,
) -> list[tuple[str, float]]:
    """Merge rankings by rank alone: each chunk scores the sum of 1 / (constant + rank).

    Ranks start at 1. A chunk both lists rank well beats one that only a
    single list ranks first. The two searches' own scores are never
    compared: BM25 scores are unbounded, cosine scores sit in a narrow high
    band (document-ingestion, "What we measured"). Equal scores keep the order
    in which chunks were first seen, so the result never depends on chance.
    """
    fused_scores: dict[str, float] = {}
    for ranked_chunk_ids in ranked_chunk_id_lists:
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (rank_constant + rank)
    # sorted() is stable, and dict order is first-seen order.
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


@dataclass(frozen=True)
class KeywordIndex:
    """BM25 over the chunks of one set of documents, in the same order as `chunks`."""

    bm25: BM25Plus
    chunks: list[StoredChunk]
    # Each chunk's keyword words, to keep only chunks that hold a query word.
    chunk_words: list[frozenset[str]]


class HybridRetriever:
    """Finds the chunks for a question in the documents asked for, by meaning and by word."""

    def __init__(
        self,
        store: DocumentStore,
        gateway: ModelGateway,
        *,
        retrieved_chunks: int = RETRIEVED_CHUNKS,
        candidates_per_search: int = CANDIDATES_PER_SEARCH,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.retrieved_chunks = retrieved_chunks
        self.candidates_per_search = candidates_per_search
        # One keyword index per set of documents, built on first use. Stored
        # documents never change, so an index built once stays correct.
        self._keyword_indexes: dict[tuple[str, ...], KeywordIndex] = {}

    async def retrieve(
        self, query: str, document_ids: list[str], *, prompt_id: str
    ) -> list[RetrievedChunk]:
        """The best chunks for one query from these documents only, best first.

        One embedding request (the query), through the gateway's embed tier
        and the run's budget, named by prompt_id in logs and traces. Each
        returned chunk's score is its fused rank score, not a similarity:
        it orders the chunks and means nothing on its own.
        """
        return await self.retrieve_for_texts([query], document_ids, prompt_id=prompt_id)

    async def retrieve_for_texts(
        self, search_texts: list[str], document_ids: list[str], *, prompt_id: str
    ) -> list[RetrievedChunk]:
        """The best chunks for several search texts at once, merged, best first (D-56).

        Each text is searched by meaning and by keyword and merged, exactly as
        for one query; the per-text rankings are then merged by the same rank
        fusion. Empty and repeated texts are dropped first, so one text gives
        the same result, scores included, as retrieve(). All texts are
        embedded in one request.
        """
        unique_document_ids = list(dict.fromkeys(document_ids))
        unique_texts = list(dict.fromkeys(text for text in search_texts if text.strip()))
        if not unique_document_ids or not unique_texts:
            return []

        text_vectors = await self.gateway.embed(unique_texts, prompt_id=prompt_id)
        chunks_by_id: dict[str, RetrievedChunk] = {}
        rankings_by_text: list[list[tuple[str, float]]] = []
        for search_text, text_vector in zip(unique_texts, text_vectors, strict=True):
            by_meaning = self.store.search(
                unique_document_ids, text_vector, k=self.candidates_per_search
            )
            by_keyword = self.keyword_search(unique_document_ids, search_text)
            for chunk in [*by_meaning, *by_keyword]:
                chunks_by_id.setdefault(chunk.chunk_id, chunk)
            rankings_by_text.append(
                reciprocal_rank_fusion(
                    [
                        [chunk.chunk_id for chunk in by_meaning],
                        [chunk.chunk_id for chunk in by_keyword],
                    ]
                )
            )
            logger.info(
                "%s: %d by meaning, %d by keyword, %d found by both, for %r",
                prompt_id,
                len(by_meaning),
                len(by_keyword),
                len({chunk.chunk_id for chunk in by_meaning} & {c.chunk_id for c in by_keyword}),
                search_text[:60],
            )

        if len(rankings_by_text) == 1:
            final_ranking = rankings_by_text[0]
        else:
            final_ranking = reciprocal_rank_fusion(
                [[chunk_id for chunk_id, _ in ranking] for ranking in rankings_by_text]
            )
        retrieved = [
            chunks_by_id[chunk_id].model_copy(update={"score": fused_score})
            for chunk_id, fused_score in final_ranking[: self.retrieved_chunks]
        ]
        logger.info(
            "%s: retrieved %d chunks for %d search texts",
            prompt_id,
            len(retrieved),
            len(unique_texts),
        )
        return retrieved

    def keyword_search(self, document_ids: list[str], query: str) -> list[RetrievedChunk]:
        """The chunks sharing words with the query, best BM25 score first.

        Only chunks that hold at least one of the query's words are returned.
        BM25Plus gives every chunk a small score, even one without any query
        word, so the filter is by the words themselves, not by score > 0. A
        query with no word in the documents gives an empty list, and the
        merge then falls back to vector search alone.
        """
        query_tokens = keyword_tokens(query)
        if not query_tokens:
            return []
        query_words = frozenset(query_tokens)
        keyword_index = self._keyword_index(document_ids)
        scores = keyword_index.bm25.get_scores(query_tokens)
        scored_positions = sorted(
            (
                position
                for position, words in enumerate(keyword_index.chunk_words)
                if query_words & words
            ),
            key=lambda position: scores[position],
            reverse=True,
        )
        return [
            RetrievedChunk(
                **keyword_index.chunks[position].model_dump(), score=float(scores[position])
            )
            for position in scored_positions[: self.candidates_per_search]
        ]

    def _keyword_index(self, document_ids: list[str]) -> KeywordIndex:
        """The BM25 index over these documents' chunks, built the first time.

        One index over all the documents asked for, not one per document:
        BM25 weighs a word by how rare it is among the chunks it searches,
        and scores from separate indexes would not be comparable.

        BM25Plus (D-53), chosen by measuring all three variants on the annual
        report, after BM25L had been chosen on a four-chunk test:

        - BM25L barely limits the reward for a repeated word: its top chunk
          for the Walmart question said "Walmart" 10 times, and the answer
          chunk, which says it twice, ranked 108th.
        - BM25Okapi and BM25Plus limit it: the answer ranked 18th for the
          question as asked and 1st for "Walmart share of total revenue
          largest customer".
        - Between those two, Okapi gives a word in half or more of the chunks
          no weight, so in a short document the very name a question asks
          about can count for nothing (on four chunks, two naming Walmart, it
          scored every chunk 0). BM25Plus keeps every word's weight above 0.
        """
        index_key = tuple(sorted(document_ids))
        if index_key not in self._keyword_indexes:
            all_chunks = [
                chunk for document_id in index_key for chunk in self.store.chunks(document_id)
            ]
            tokenised_chunks = [keyword_tokens(chunk.text) for chunk in all_chunks]
            self._keyword_indexes[index_key] = KeywordIndex(
                bm25=BM25Plus(tokenised_chunks),
                chunks=all_chunks,
                chunk_words=[frozenset(words) for words in tokenised_chunks],
            )
        return self._keyword_indexes[index_key]
