# AGENTS

For local validation in this repository, use `python run_test.py <test-name>` as the standard command.

For semantic convention capturability reviews, especially manual instrumentation examples for native library instrumentation, use the `capturability-study` skill under `.github/skills/capturability-study/`.

Optimize all code in this repository for readability and simplicity.

- Avoid advanced syntax when an equivalent simpler form is available.
- Prefer straightforward control flow and explicit names over dense or compact constructs.
- Let errors bubble up and fail loudly. Do not swallow exceptions with try/except.
