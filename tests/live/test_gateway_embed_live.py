"""Live check of the gateway's embed tier against the real Mistral API.

CONCEPT: document-ingestion, decision 3. Runs only on request
(`uv run pytest -m live`), never in CI. One request of three short texts:
it confirms the real client, built with its own retry off and the local
tokenizer, returns one 1,024-number vector per text, and that texts about
the same thing land closer together than texts about different things.
"""

import math
import time

import pytest

from graph.llm import ModelGateway

pytestmark = pytest.mark.live

MISTRAL_EMBED_DIMENSIONS = 1024

WAREHOUSE_TEXT = "Symbotic sells robotic systems that automate warehouse storage and retrieval."
SIMILAR_TEXT = "The company builds robots that move cases around distribution centres."
UNRELATED_TEXT = "Islamabad is the capital city of Pakistan."


def cosine_similarity(first_vector: list[float], second_vector: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first_vector, second_vector, strict=True))
    first_length = math.sqrt(sum(a * a for a in first_vector))
    second_length = math.sqrt(sum(b * b for b in second_vector))
    return dot_product / (first_length * second_length)


async def test_embed_tier_returns_one_vector_per_text_from_mistral():
    gateway = ModelGateway()
    started_at = time.perf_counter()

    warehouse_vector, similar_vector, unrelated_vector = await gateway.embed(
        [WAREHOUSE_TEXT, SIMILAR_TEXT, UNRELATED_TEXT], prompt_id="LIVE-EMBED"
    )

    elapsed_seconds = time.perf_counter() - started_at
    similar_score = cosine_similarity(warehouse_vector, similar_vector)
    unrelated_score = cosine_similarity(warehouse_vector, unrelated_vector)
    print(
        f"\nembed tier: 3 vectors of {len(warehouse_vector)} in {elapsed_seconds:.1f} s; "
        f"similar {similar_score:.3f}, unrelated {unrelated_score:.3f}"
    )
    assert len(warehouse_vector) == MISTRAL_EMBED_DIMENSIONS
    assert similar_score > unrelated_score
