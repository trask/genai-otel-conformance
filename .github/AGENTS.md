# AGENTS

For local validation in this repository, use `python run_test.py <test-name>` as the standard command.

For semantic-conventions PR work that needs repository-wide prototype coverage, use the `prototype` skill under `.github/skills/prototype/`.

For reviews of resulting prototype coverage, capturability, and honest capture gaps, use the `evaluate-prototype` skill under `.github/skills/evaluate-prototype/`.

Optimize all code in this repository for readability and simplicity.

- Avoid advanced syntax when an equivalent simpler form is available.
- Prefer straightforward control flow and explicit names over dense or compact constructs.
- Let errors bubble up and fail loudly. Do not swallow exceptions with try/except.
