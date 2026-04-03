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

import json
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.io_artifacts import (
    JsonObject,
    RowLabeler,
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax, cross_entropy_loss

__all__ = [
    "make_training_pairs",
    "row_labeler_context2",
    "token_row_index_context2",
    "train_model",
]

type Context2 = tuple[int, int]
type Context2Pair = tuple[Context2, int]

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

LOG: logging.Logger = get_logger("TRAIN", level="INFO")


def token_row_index_context2(context_ids: Context2, vocab_size: int) -> int:
    """Return the row index for a context-2 token sequence.

    We need to map a 2D context (previous, current) to a 1D row index.
    This is like converting 2D coordinates to a 1D array index.

    Example with vocab_size=10:
        context (0, 0) -> row 0
        context (0, 1) -> row 1
        context (0, 9) -> row 9
        context (1, 0) -> row 10
        context (1, 1) -> row 11
        context (2, 5) -> row 25

    Formula:
        row_index = token_id_{t-1} * vocab_size + token_id_t

    This creates vocab_size * vocab_size unique rows,
    one for each possible (previous, current) pair.
    """
    token_id_t_minus_1, token_id_t = context_ids
    return token_id_t_minus_1 * vocab_size + token_id_t


def row_labeler_context2(vocab: VocabularyLike, vocab_size: int) -> RowLabeler:
    """Map a context-2 row index back to a readable label like 'the|cat'.

    This reverses the flattening done by token_row_index_context2().

    Example with vocab_size=10:
        row 0  -> (0, 0) -> "tok_0|tok_0"
        row 25 -> (2, 5) -> "tok_2|tok_5"

    The math (integer division and modulo) undoes the flattening:
        token_id_{t-1} = row_index // vocab_size  (which "block" of vocab_size)
        token_id_t     = row_index % vocab_size   (position within that block)
    """

    def label(row_idx: int) -> str:
        # Reverse the flattening: row_index -> (token_id_{t-1}, token_id_t)
        token_id_t_minus_1: int = row_idx // vocab_size
        token_id_t: int = row_idx % vocab_size

        # Convert token IDs to readable strings
        token_ids: list[int] = [token_id_t_minus_1, token_id_t]
        tokens: list[str] = [
            vocab.get_id_token(tid) or f"id_{tid}" for tid in token_ids
        ]

        return "|".join(tokens)

    return label


def make_training_pairs(token_ids: list[int]) -> list[Context2Pair]:
    """Convert token IDs into ((t-1, t), next) training pairs.

    Example:
        ids = [3, 1, 2, 4]
        pairs = [((3, 1), 2), ((1, 2), 4)]
    """
    pairs: list[Context2Pair] = []
    for i in range(len(token_ids) - 2):
        context_ids: Context2 = (token_ids[i], token_ids[i + 1])
        next_id: int = token_ids[i + 2]
        pairs.append((context_ids, next_id))
    return pairs


def train_model(
    model: "SimpleNextTokenModel",
    pairs: list[Context2Pair],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the model using gradient descent on softmax cross-entropy (context-2).

    Training proceeds in epochs (full passes through all training pairs).
    For each pair, we:
    1. Compute the model's predicted probabilities (forward pass).
    2. Measure how wrong the prediction was (loss).
    3. Adjust weights to reduce the loss (gradient descent).

    Each example:
        context_ids = (token_id_{t-1}, token_id_t)
        target_id   = token_id_{t+1}

    Args:
        model: The model to train (weights will be modified in place).
        pairs: List of Context2Pair training pairs.
        learning_rate: Step size for gradient descent. Larger values learn
            faster but may overshoot; smaller values are more stable but slower.
        epochs: Number of complete passes through the training data.

    Returns:
        List of per-epoch metrics dictionaries containing epoch number,
        average loss, and accuracy.
    """
    history: list[dict[str, float]] = []
    vocab_size: int = model.vocab_size

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for context_ids, target_id in pairs:
            previous_id, current_id = context_ids

            # Forward pass: probabilities for next token given (t-1, t).
            probs: list[float] = model.forward(previous_id, current_id)

            # Loss: cross-entropy for the correct next token.
            loss: float = cross_entropy_loss(probs, target_id)
            total_loss += loss

            # Accuracy: did the top prediction match the target?
            pred_id: int = argmax(probs)
            if pred_id == target_id:
                correct += 1

            # Backward pass: compute gradients and update weights.
            #
            # For softmax cross-entropy, the gradient has an elegant form:
            #   gradient[j] = predicted_prob[j] - true_prob[j]
            #
            # Since true_prob is one-hot (1.0 for target, 0.0 elsewhere):
            #   - For the target token: gradient = prob - 1.0 (negative, so weight increases)
            #   - For other tokens: gradient = prob - 0.0 (positive, so weight decreases)
            #
            # This pushes probability mass toward the correct token.
            # Update the weight row for this specific (t-1, t) context.
            row_idx: int = token_row_index_context2(context_ids, vocab_size=vocab_size)
            row: list[float] = model.weights[row_idx]

            for j in range(vocab_size):
                y: float = 1.0 if j == target_id else 0.0
                grad: float = probs[j] - y
                row[j] -= learning_rate * grad

        # Compute epoch-level metrics.
        avg_loss: float = total_loss / len(pairs) if pairs else float("nan")
        accuracy: float = correct / len(pairs) if pairs else 0.0

        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "avg_loss": avg_loss,
            "accuracy": accuracy,
        }
        history.append(metrics)

        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run a simple training demo end-to-end."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Softmax Regression")

    base_dir: Final[Path] = Path(__file__).resolve().parents[2]
    outputs_dir: Final[Path] = base_dir / "outputs"
    train_log_path: Final[Path] = outputs_dir / "train_log.csv"

    # Step 0: Identify the corpus file (single file rule).
    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
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
    write_training_log(train_log_path, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=base_dir,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="context2",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_context2(vocab, vocab_size),
    )

    # 7c: After write_artifacts(), append start_tokens to meta
    meta_path: Path = base_dir / "artifacts" / "00_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta: JsonObject = json.load(f)
    start_tokens: list[str] = tokens[:2]  # context-2
    meta["start_tokens"] = list(start_tokens)  # explicit list for JSON serialization
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

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
