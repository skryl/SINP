# SINP

Structure and interpretation of neural programs.

An interactive, notebook-native textbook that teaches neural networks and large-language models from first principles—modeled after Structure and Interpretation of Computer Programs (SICP) but written in modern Python.

"Programs must be written for people to read, and only incidentally for machines to execute." — Abelson & Sussman (adapted for gradients by SINP)

---

## What You'll Find Here

| Folder | Purpose |
| --- | --- |
| part01_building_abstractions/ | Foundational math, tensors, & automatic differentiation |
| part02_state_and_modularity/ | Parameter objects, optimizers, data streams, a micro-framework |
| part03_transformer_interpreter/ | Attention, tokenization, encoder/decoder models |
| part04_scaling_up/ | Mixed precision, distributed training, scaling laws, RAG |
| part05_reflection_and_ethics/ | Probing, alignment, serving, open problems |
| assets/ | SVG diagrams & printable figures |
| environments/ | Conda & Dev-container specs for reproducible installs |
| tests/ | pytest + nbval suites that keep notebooks executable |
| examples/ | Finished capstone projects contributed by the community |

Each chapter lives in its own numbered notebook, ending with executable exercises and starred research prompts (★).

---

## Quick Start (5 minutes)

```bash
# 1 Clone the repo
git clone https://github.com/<your_org>/sinp.git && cd sinp

# 2 Create the environment (Conda)
conda env create -f environments/environment.yml
conda activate sinp

# 3 Launch JupyterLab
jupyter lab
```

Prefer VS Code Dev Containers?

```bash
docker compose up
# or “Dev Containers: Open Folder in Container” in VS Code
```

Note: The default environment targets Python ≥ 3.11 with CUDA 11.8. CPU-only? Add `DEVICE=cpu` before running `docker compose`.

---

## Book Outline

### Preface

- Why another NN/LLM book?
- How to use this text & the Jupyter/Colab notebook set
- Suggested pacing for self-study, university semesters, and intensive boot-camps
- Conventions & a quick tour of Python ≥ 3.11, pytest, and ruff for literate tests

---

### Part I — Building Abstractions with Functions & Data

| Ch | Title | Core Ideas | Notebook Lab |
| --- | --- | --- | --- |
| 1 | A Pythonic Calculus of Procedures | Pure vs. stateful functions, higher-order funcs, closures | 01_hof_basics.ipynb – build compose, curry, simple REPL |
| 2 | Numbers with Errors | Rational, complex, interval & automatic differentiation (dual numbers) | 02_interval_dual.ipynb – exact rational class; forward AD for sin/cos |
| 3 | Vectors → Tensors as Abstract Data | Tensor shape algebra, broadcasting, memory layout | 03_tensor_algebra.ipynb – minimal NumPy-free tensor class |
| 4 | The Meta-Differentiable Evaluator | Reverse-mode AD via an interpreter that records elementary ops | 04_autograd_from_scratch.ipynb – tiny Autograd clone |

*SICP Resonance:* These chapters mirror SICP’s generic arithmetic & meta-circular evaluator, but with differentiation as the unifying abstraction.

---

### Part II — Modularity, Objects, & State in Learning Systems

| Ch | Title | Core Ideas | Notebook Lab |
| --- | --- | --- | --- |
| 5 | Parameter Objects & Mutation | Why learning needs mutable state; SGD as controlled side-effect | 05_sgd_optimizer.ipynb – write dense layer & vanilla SGD |
| 6 | Sequences, Recursion, & Scan Ops | RNNs via higher-order scan; timestep unrolling vs. symbolic gradients | 06_rnn_scan.ipynb – build a char-level RNN from scratch |
| 7 | Mini-Framework: tinygrad-SICP | Assemble modules into a micro-framework (<200 LOC) | 07_framework.ipynb – CIFAR-10 training in under 1 CPU-hour |
| 8 | Streams, Generators, & Data Pipes | Lazy evaluation for infinite datasets, TFRecord/Parquet ingestion | 08_streams.ipynb – build sharded token stream with Python generators |

---

### Part III — Metalinguistic Abstraction: The Transformer as an Interpreter

| Ch | Title | Core Ideas | Notebook Lab |
| --- | --- | --- | --- |
| 9 | From Attention to Self-Interpretation | Dot-product attention as a dispatch table | 09_attention.ipynb – visualize key/query overlap heatmaps |
| 10 | Building a Transformer Encoder | Multi-head, layer norm, residual pathways | 10_transformer_encoder.ipynb – train tiny BERT on toy corpus |
| 11 | Byte-Pair Encoding & Token Semirings | Compression as abstraction; implement BPE, unigram, WordPiece | 11_tokenizer.ipynb – learn BPE rules from Shakespeare |
| 12 | Autoregressive Decoders & Causal Masks | Parallel scan, KV-cache, temperature sampling | 12_decoder.ipynb – sampler with nucleus & repetition penalties |

---

### Part IV — Scaling Up: From Toy Models to LLMs

| Ch | Title | Core Ideas | Notebook Lab |
| --- | --- | --- | --- |
| 13 | Optimizers Beyond SGD | Adam, Adafactor, Lion—derive from first principles | 13_optimizer_family.ipynb – compare on MNIST-1D |
| 14 | Mixed Precision & Hardware Kernels | FP8/FP16 numerics, CUDA kernels, intro to triton | 14_fused_kernels.ipynb – write a fused softmax in Triton |
| 15 | Distributed Data & Model Parallel | Pipeline vs. tensor vs. expert parallel | 15_distributed_training.ipynb – run DDP on 2-GPU simulation |
| 16 | Scaling Laws & Curriculum | Empirical power laws, Chinchilla optimality | 16_scaling_laws.ipynb – reproduce Kaplan curves with synthetic data |
| 17 | In-Context Learning & Function Calling | Prompt engineering, retrieval augmentation | 17_icl_tools.ipynb – build a simple RAG pipeline |

---

### Part V — Reflection, Interpretation, & Ethics

| Ch | Title | Core Ideas | Notebook Lab |
| --- | --- | --- | --- |
| 18 | Analyzing Internal Representations | Probing classifiers, causal tracing | 18_rep_probing.ipynb – detect neuron directions for sentiment |
| 19 | Alignment & Value Learning | RLHF, Constitutional AI, interpretability tooling | 19_rlhf_minimal.ipynb – preference-based RL on toy tasks |
| 20 | Systems Engineering for Production | Serving, quantization, streaming, cold-start latency | 20_serving.ipynb – deploy a quantized model with FastAPI |
| 21 | Limits, Open Problems, & Next Steps | Non-IID data, continual learning, neurosymbolic hybrids | 21_future.ipynb – experiment: hybrid transformer + SAT solver |

---

### Epilogue

- The meta-circular gradient: deriving AD inside an LLM
- Further reading: DSP, Deep Learning, PLDI transformer papers
- Historical timeline from McCulloch & Pitts (1943) to GPT‑4o (2024)

---

### Pedagogical Features (SICP-Inspired)

1. Executable Exercises – numbered tasks; ★ problems are open-ended research prompts.
2. Orange Boxes → “Foot-Guns Ahead” – common implementation pitfalls.
3. Purple Boxes → “Abstraction Barriers” – highlight intentional boundary crossings.
4. Capstone Projects – e.g., build a 70 M-parameter GPT-Mini on TinyStories.
5. Meta-Circular Threads – recurring theme: “a learner that can explain its own gradient”.
6. Interleaved Essays – ethics, scaling economics, hardware trends.

---

### Getting Started

1. Open `part01_building_abstractions/01_hof_basics.ipynb` in JupyterLab.
2. Execute cells top-to-bottom (`Shift+Enter`).
3. Complete the exercises—answers are validated by hidden tests.

Hints & solutions live in collapsible Spoilers cells.

---

### Tested Configurations

| Platform | Python | GPU | Status |
| --- | --- | --- | --- |
| Ubuntu 22.04 | 3.11 / 3.12 | RTX 4090 (CUDA 11.8) | ✅ all parts |
| macOS 14 (Apple Silicon) | 3.11 | M-series (Metal) | ✅ Parts I–III |
| Windows 11 WSL2 | 3.11 | RTX 30xx | ✅ tested |

CI runs on GitHub Actions for CPU notebooks; GPU notebooks are smoke-tested nightly.

---

### Contributing

1. Fork ➡ Branch ➡ PR.
2. Run `ruff check . --fix` and `pytest -q`; both must pass.
3. For new notebooks, add three automated tests in `tests/`.
4. Agree to the lightweight Contributor License Agreement in your PR description.

We love bug reproductions, exercise ideas, and real-world datasets for capstones.

---

### License

© 2025 MIT License—see `LICENSE` for details. Diagrams under CC-BY 4.0.

Commercial use? Reach out via Issues.

---

### Citation

```bibtex
@misc{skryl2025sinp,
  title  = {Structure and Interpretation of Neural Programs},
  author = {Alex Skryl and Contributors},
  year   = {2025},
  url    = {https://github.com/<your_org>/sinp}
}
```

---

### Acknowledgements

- Gerald Jay Sussman & Hal Abelson for SICP’s timeless pedagogy.
- Inspirations: tinygrad, micrograd, minGPT, nanoGPT, lit‑gpt.
- Early testers in the #sinp-alpha Discord.

Happy gradient-hacking!
