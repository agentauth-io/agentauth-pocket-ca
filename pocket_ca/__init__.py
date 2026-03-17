"""Shared utilities for the AgentAuth Pocket CA pipeline.

The ``pocket_ca`` package contains the core logic that is shared across
data generation, training, and evaluation stages:

* ``data_utils`` -- Record loading, normalisation, deduplication, and
  stratified train/val/test splitting.
* ``formatting`` -- Llama 3 chat-template prompt construction, decision
  label normalisation (approve/reject/review), and response parsing.
* ``eval_utils`` -- Model + LoRA checkpoint loading and deterministic
  inference helpers for evaluation scripts.
"""
