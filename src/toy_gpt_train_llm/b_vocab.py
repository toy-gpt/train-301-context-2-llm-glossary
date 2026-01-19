"""b_vocab.py - Vocabulary construction module.

Builds a vocabulary from tokenized text data.

Responsibilities:
- Identify unique tokens in the corpus
- Assign each token a unique integer ID
- Compute token frequency counts
- Provide token-to-id and id-to-token mappings

This module bridges raw tokens and numerical representations
required by statistical and neural models.
"""

import logging

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.b_vocab import Vocabulary

__all__ = ["Vocabulary"]

LOG: logging.Logger = get_logger("VOCAB", level="INFO")


def main() -> None:
    """Demonstrate vocabulary construction from the project corpus."""
    log_header(LOG, "Vocabulary Demo")

    # Local import keeps modules decoupled and avoids import work unless needed.
    from toy_gpt_train_llm.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer

    log_header(LOG, "Vocabulary Demo")

    # 1) Tokenize - start with corpus and turn it into tokens
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    # 2) Build vocabulary (the set of unique tokens) from the list of tokens
    vocab = Vocabulary(tokens)
    LOG.info(f"Vocabulary size: {vocab.vocab_size()}")

    if tokens:
        sample_token = tokens[0]
        sample_id = vocab.get_token_id(sample_token)
        sample_freq = vocab.get_token_frequency(sample_token)

        LOG.info(
            f"Sample token: {sample_token!r} "
            f"| ID: {sample_id} "
            f"| Frequency: {sample_freq}"
        )
    else:
        LOG.info("No tokens found; cannot demonstrate vocabulary lookup.")


if __name__ == "__main__":
    main()
