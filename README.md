# TrueSlice

This repository contains code forked from revision `6b12cde` of the SliceGPT codebase.

- `src/trueslice` contains the TrueSlice implementation.
- `src/slicegpt` contains an altered version of SliceGPT used as a baseline for TrueSlice. These alterations update compatibility with Hugging Face Transformers and add support for newer model families. Both implementations currently support Transformers **5.2.0**.
- `research_utils` contains utilities for running experiments and calculating the method-specific sparsity inputs needed for experiments with matched parameter counts.

The original tests have not yet been updated and are incompatible with both TrueSlice and the updated SliceGPT implementation. We plan to update them when time permits.

Upstream copyright and license notices are retained. See `LICENSE` and the notices in individual source files.
