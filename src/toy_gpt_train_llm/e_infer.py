"""e_infer.py - Inference module (artifact-driven).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load inspectable training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv
- Reconstruct a vocabulary-like interface and model weights
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection

Notes:
- This module does NOT retrain by default.
- If artifacts are missing, run d_train.py first.

- Context-2 bootstrapping: generation starts from a single start token.
  To form the first 2-token context, we use (start, start) as the initial context.
"""

import argparse
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.e_infer import (
    ArtifactVocabulary,
    generate_tokens_context2,
    load_meta,
    load_model_weights_csv,
    load_vocabulary_csv,
    require_artifacts,
    top_k,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

LOG: logging.Logger = get_logger("INFER", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"
META_PATH: Final[Path] = ARTIFACTS_DIR / "00_meta.json"
VOCAB_PATH: Final[Path] = ARTIFACTS_DIR / "01_vocabulary.csv"
WEIGHTS_PATH: Final[Path] = ARTIFACTS_DIR / "02_model_weights.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Toy GPT inference from saved artifacts."
    )
    parser.add_argument(
        "--start",
        dest="start_token",
        default="",
        help="Start token for generation. If omitted, uses the first token in the vocabulary.",
    )
    parser.add_argument(
        "--num",
        dest="num_tokens",
        type=int,
        default=10,
        help="Number of tokens to generate (not counting the start token).",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=3,
        help="Show top-k next-token probabilities for the start token.",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text")

    require_artifacts(
        meta_path=META_PATH,
        vocab_path=VOCAB_PATH,
        weights_path=WEIGHTS_PATH,
        train_hint="uv run python src/toy_gpt_train/d_train.py",
    )

    meta: JsonObject = load_meta(META_PATH)
    vocab: ArtifactVocabulary = load_vocabulary_csv(VOCAB_PATH)

    v: int = vocab.vocab_size()
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=v)
    model.weights = load_model_weights_csv(
        WEIGHTS_PATH,
        vocab_size=v,
        expected_rows=v * v,
    )

    args: argparse.Namespace = parse_args()

    # Choose a start token.
    start_token = args.start_token
    if not start_token:
        # Deterministic fallback: smallest token_id present
        first_id = min(vocab.id_to_token.keys())
        start_token = vocab.id_to_token[first_id]

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {v}")
    LOG.info(f"Start token: {start_token}")
    LOG.info(f"Context-2 bootstrap: ({start_token}, {start_token})")

    start_id = vocab.get_token_id(start_token)
    if start_id is not None:
        probs: list[float] = model.forward(start_id, start_id)
        LOG.info(f"Top next-token predictions after {start_token}|{start_token}:")
        for tok_id, prob in top_k(probs, k=max(1, args.topk)):
            tok = vocab.get_id_token(tok_id)
            LOG.info(f"  {tok} (ID {tok_id}): {prob:.4f}")

    generated = generate_tokens_context2(
        model=model,
        vocab=vocab,
        start_token=start_token,
        num_tokens=max(0, args.num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
