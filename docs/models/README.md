# Browser AI assets

The generated model and embedding index live here after the GitHub Action runs:

- `cuneiform-encoder.onnx`
- `sign-index.json`
- `sign-embeddings.f32`

They are intentionally generated in GitHub Actions and placed directly in the GitHub Pages
deployment artifact, not committed to the repository. The user's computer therefore does
not need PyTorch or the training toolchain and does not store the generated assets.

The initial encoder uses ImageNet weights and is only a plumbing prototype; it must be
fine-tuned on cuneiform sketches before its rankings are treated as research results.
