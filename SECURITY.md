# Security Policy

## Supported versions

Only the latest state of the `Production` branch receives security fixes.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub:
[Security → Report a vulnerability](https://github.com/Deatrix09/whisper-voice-transcriber/security/advisories/new).

Please include a description of the issue, steps to reproduce and the affected
component (CLI, Streamlit UI, Docker image, config parsing, …). You can expect an
initial response within 7 days. Once a fix is available, the advisory will be
published with credit to the reporter (unless you prefer to stay anonymous).

## Handling secrets

The application only needs a HuggingFace token for the optional `pyannote`
diarization module. Keep it in a local `.env` file (see `.env.example`) or an
environment variable — never in `config/*.yaml` or in commits.
