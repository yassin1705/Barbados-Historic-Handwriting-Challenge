# Contributing

Contributions are welcome, especially reproducibility fixes, tests, and validation methods that do not leak competition data.

1. Create a focused branch.
2. Keep data, model weights, adapters, predictions, submissions, and executed notebook output out of commits.
3. Run `pytest` and `python scripts/check_public_notebooks.py`.
4. Describe any change to the split, preprocessing, prompt, metric, or checkpoint selection because it can invalidate comparisons.

Do not open an issue or pull request containing competition samples, labels, credentials, or private paths.

