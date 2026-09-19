# Verification scope

- 30 Python files parse successfully, including the two added CLI/helper test files.
- 18 isolated checks pass: both entry-point help commands and imports with Python `-S`, explicit true/false values, invalid boolean rejection, environment dataset defaults, and CLI overrides.
- AST comparison confirms training/evaluation statements after parsing are unchanged apart from moving imports and setting GPU visibility before PyTorch import. Other existing source files and published images match the verified GitHub base.
- The 35-file server archive matches the existing GitHub content after line-ending normalization. All 19 pre-existing commits are retained locally.

Run `python -m pip install -r requirements-check.txt` and `python -m pytest -q tests` from the repository root. The checks use no model dependency, network, real images, patient records or checkpoints. They do not run training, model inference, GPU workflows or validate research metrics. Known reproduction limitations are listed in ORGANIZATION.md.
