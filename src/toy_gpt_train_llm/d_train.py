"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using a context-2 window (two tokens of context).

Responsibilities:
- Create ((token_{t-1}, token_t) -> next_token) training pairs
- Run a basic gradient-descent training loop
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- context-2: predict the next token using (previous token, current token)
- epoch: one complete pass through all training pairs
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct next token
- gradient descent: iterative weight updates to reduce prediction error
  - think descending to find the bottom of a valley in a landscape
  - where the valley floor corresponds to lower prediction error

Notes:
- This remains intentionally simple: no deep learning framework, no Transformer.
- The model generalizes n-gram training by expanding the context window.
- Training updates weight rows associated with the observed context-2 pattern.
- token_embeddings.csv is a visualization-friendly projection for levels 100-400;
  in later repos (500+), embeddings become a first-class learned table.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.d_train import (
    make_training_pairs,
    row_labeler_context2,
    train_model,
)
from toy_gpt_train.math_training import argmax

from toy_gpt_train_llm.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
from toy_gpt_train_llm.b_vocab import Vocabulary
from toy_gpt_train_llm.io_artifacts import (
    write_artifacts,
    write_training_log,
)

type Context2 = tuple[int, int]
type Context2Pair = tuple[Context2, int]

LOG: logging.Logger = get_logger("TRAIN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"


def main() -> None:
    """Run a simple training demo end-to-end."""
    log_header(LOG, "Training Demo: Next-Token Softmax Regression")

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < 3:
        LOG.error("Need at least 3 tokens for context-2 training (t-1, t -> next).")
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)
    vocab_size: int = vocab.vocab_size()

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error(f"Token not found in vocabulary: {tok}")
            return
        token_ids.append(tok_id)

    # Step 4: Create training pairs (context-2 -> next).
    pairs: list[Context2Pair] = make_training_pairs(token_ids)
    LOG.info(f"Created {len(pairs)} training pairs.")

    # Step 5: Initialize model with zero weights (context-2 table lives in c_model.py).
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab_size)

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(TRAIN_LOG_PATH, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=DEFAULT_CORPUS_PATH,
        vocab=vocab,
        model=model,
        model_kind="context2",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_context2(vocab, vocab_size),
    )

    # Step 8: Qualitative check - what does the model predict after the first 2 tokens?
    previous_token: str = tokens[0]
    current_token: str = tokens[1]
    previous_id: int | None = vocab.get_token_id(previous_token)
    current_id: int | None = vocab.get_token_id(current_token)
    if previous_id is None or current_id is None:
        LOG.error("One of the sample tokens was not found in vocabulary.")
        return

    probs: list[float] = model.forward(previous_id, current_id)
    best_next_id: int = argmax(probs)
    best_next_tok: str | None = vocab.get_id_token(best_next_id)

    LOG.info(
        f"After training, most likely next token after {previous_token!r}|{current_token!r} "
        f"is {best_next_tok!r} (ID: {best_next_id})."
    )


if __name__ == "__main__":
    main()
