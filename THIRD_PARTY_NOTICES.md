# Third-party notices

The original source code and documentation in this repository are distributed
under the repository's [`LICENSE`](LICENSE). External datasets, model weights,
container images, runtime packages, and browser libraries remain governed by
their respective publishers' licenses, model cards, terms, and notices.

## Data and model assets

The repository does not redistribute the following assets. Acquisition code
resolves and records immutable revisions at runtime:

- `roneneldan/TinyStories` supplies training text, validation text, and the
  published prompt set;
- `roneneldan/TinyStories-8M` is the primary external model;
- `roneneldan/TinyStories-33M` is the larger capacity anchor;
- `roneneldan/TinyStories-1M` and `roneneldan/TinyStories-3M` may be used by
  the optional official-model matrix;
- `Rowan/hellaswag` supplies the secondary HellaSwag diagnostic.

These resources are downloaded from Hugging Face. Users must review the dataset
cards, model cards, repository files, and current terms before downloading or
redistributing any asset. A generated local cache or handoff must not be
mistaken for a grant to redistribute upstream content.

## Containers and machine-learning runtimes

The DGX Dockerfile references two external images:

- `nvcr.io/nvidia/pytorch:25.11-py3`, supplied through NVIDIA NGC, provides
  PyTorch, CUDA, and the DGX machine-learning runtime;
- `node:22.16.0-bookworm-slim` supplies the Node.js executable and npm tooling
  copied into the derived image.

The image names are build inputs, not files distributed in this source tree.
Use of NGC, NVIDIA software, the Node.js image, Docker base layers, and any
derived container is subject to the applicable publisher terms. An official
experiment must resolve and retain the real NGC registry RepoDigest rather than
treating the mutable tag as an immutable identity.

On the commodity-x86 host, `npm run setup -- --locked-cpu` downloads the
official CPU-only `torch==2.10.0` wheel from the PyTorch package index. PyTorch
is not vendored by this repository and is intentionally not listed in
`requirements-dgx.txt`, because the DGX image owns its PyTorch/CUDA stack.

## Python and documentation dependencies

Direct Python experiment dependencies are declared in `pyproject.toml` and
exactly pinned for the official DGX/x86 workflow in `requirements-dgx.txt`.
They include NumPy, SentencePiece, PyYAML, psutil, Hugging Face Hub,
Transformers, Datasets, Safetensors, and pytest. Build tooling includes
setuptools, pip, and wheel. These packages are installed from their publishers;
their own license texts and transitive dependencies apply.

The HTML documentation loads Mermaid 11 as an ES module from jsDelivr when a
page is opened with network access. Mermaid and jsDelivr content are not
vendored in the repository. The technical sources page at
[`docs/sources.html`](docs/sources.html) identifies the research papers,
datasets, models, runtime documentation, and implementation references used by
the protocol.

Before publishing a derived archive, inspect its actual contents and preserve
all notices required by any externally obtained files that the archive adds.
