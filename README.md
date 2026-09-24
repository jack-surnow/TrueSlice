# TrueSlice

This repository contains code forked from revision `6b12cde` of the SliceGPT codebase  (https://github.com/microsoft/TransformerCompression).

- `src/trueslice` contains the TrueSlice implementation.
- `src/slicegpt` contains an altered version of SliceGPT used as a baseline for TrueSlice. These alterations update compatibility with Hugging Face Transformers and add support for newer model families. Both implementations currently support Transformers **5.2.0**.
- `research_utils` contains utilities for running experiments and calculating the method-specific sparsity inputs needed for experiments with matched parameter counts.

Supported model families:

- **SliceGPT:** OPT (excluding OPT-350M), Phi-2, Phi-3 Mini (4K Instruct), Llama 2, Llama 3, Llama 3.1, Llama 3.2, and Qwen3.
- **TrueSlice:** Phi-3 Mini (4K Instruct), Llama 2, Llama 3, Llama 3.1, Llama 3.2, Qwen3, and Qwen3.5. OPT (excluding OPT-350M) and Phi-2 can also be used, but TrueSlice is not designed for these architectures and incurs additional perplexity and task performance degradation.

The original tests have not yet been updated and are incompatible with both TrueSlice and the updated SliceGPT implementation. We plan to update them when time permits.

Upstream copyright and license notices are retained. See `LICENSE` and the notices in individual source files.

## Installation and usage

From the repository root, install the package and experiment dependencies:

```bash
python -m pip install -e '.[experiment]'
```

The scripts in `research_utils` provide example configurations used during development. Before running them, adjust the model selections, output paths, and experiment settings for your environment.

For example, to run the configured compression experiments:

```bash
./research_utils/slice_model.sh
```

This script may run multiple models and sparsity settings. It should be run from the repository root.