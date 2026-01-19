"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for a context-2 setting
(uses two tokens in sequence as context).

Responsibilities:
- Represent a simple parameterized model that maps a
  2-tuple of token IDs (previous token, current token)
  to a score for each token in the vocabulary.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

Compare context-2 and bigram model (train-200):
- Both models use the same mathematical structure:
  a conditional distribution p(next | previous, current).
- Both models use a weight table conceptually shaped as:
    prev x curr x next
  (flattened for storage).

The difference is conceptual, not mathematical:
- The bigram model (train-200) is presented as a classical n-gram idea,
  emphasizing conditional next-token statistics.
- The context-2 model (train-300) reframes the same structure as a
  sliding context window, laying the foundation for:
    - context-3 models (train-400)
    - embeddings (train-500)
    - attention mechanisms (train-600)

Conceptually:
- Bigram answers: "What usually comes next after this pair?"
- Context-2 answers: "Given the recent local context, what comes next?"
- Results should be identical between the two models.
- The context-2 framing is more extensible for future models.

This model is intentionally simple:
- one weight table (conceptually a 3D tensor: prev x curr x next,
  flattened for storage)
- one forward computation
- no learning here

Training is handled in a different module.
"""

import logging

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel

__all__ = ["SimpleNextTokenModel"]

LOG: logging.Logger = get_logger("MODEL", level="INFO")


def main() -> None:
    """Demonstrate a forward pass of the simple context-2 model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train_llm.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
    from toy_gpt_train_llm.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo (Context-2)")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < 2:
        LOG.info("Need at least two tokens for context-2 demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Select context tokens (previous, current).
    previous_token: str = tokens[1]
    current_token: str = tokens[2]

    previous_id: int | None = vocab.get_token_id(previous_token)
    current_id: int | None = vocab.get_token_id(current_token)

    if previous_id is None or current_id is None:
        LOG.info("One of the sample tokens was not found in vocabulary.")
        return

    # Step 5: Forward pass (context-2).
    probs: list[float] = model.forward(previous_id, current_id)

    # Step 6: Inspect results.
    LOG.info(
        f"Input tokens: {previous_token!r} (ID {previous_id}), {current_token!r} (ID {current_id})"
    )
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
