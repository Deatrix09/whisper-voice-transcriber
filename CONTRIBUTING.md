# Contributing

Thanks for your interest in improving **whisper-voice-transcriber**! Bug reports,
ideas and pull requests are all welcome.

## Ground rules

- `Production` is the only long-lived branch and it is protected: **all changes land
  through pull requests**. Direct pushes are reserved for the maintainer.
- Fork the repository, create a topic branch in your fork
  (e.g. `fix/vad-empty-chunk`, `feat/srt-speaker-colors`) and open a PR against `Production`.
- Every PR needs a green `tests` check and an approving review from the maintainer
  (see [CODEOWNERS](.github/CODEOWNERS)) before it can be merged.
- Keep PRs focused: one logical change per PR is much easier to review.
- By submitting a contribution you agree that it is licensed under the
  [Apache License 2.0](LICENSE), the same license as the project.

## Development setup

```bash
git clone https://github.com/<your-user>/whisper-voice-transcriber.git
cd whisper-voice-transcriber
uv sync --extra test            # or: python -m venv venv && pip install -r requirements-cpu.txt pytest
```

You also need **FFmpeg** on `PATH` (see the README). A GPU is not required for development:
the test suite mocks the heavy models and runs on CPU in well under a minute.

## Before opening a PR

```bash
python -m pytest tests/ -q
python -m compileall -q src algorithm
```

- Add or update tests for the behaviour you change (`tests/`).
- New source files start with the license header used across the project:

  ```python
  # Copyright (c) 2024-2026 Kleveta Lukas
  # SPDX-License-Identifier: Apache-2.0
  ```

- Update `README.md` / `config/config.yaml` comments if you add or change options.

## Never commit

- Tokens or secrets (`HF_TOKEN` belongs in a local `.env`, see `.env.example`).
- Audio recordings, transcripts or other personal / copyrighted data. Use
  `data/input/` locally — it is git-ignored — and generate tiny synthetic WAVs in tests
  (see `tests/conftest.py`).
- Model weights or other large binaries.

## Reporting bugs and security issues

- Bugs and feature requests: use the [issue templates](https://github.com/Deatrix09/whisper-voice-transcriber/issues/new/choose).
- Security vulnerabilities: please follow [SECURITY.md](SECURITY.md) and report privately.

Please be respectful — this project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
