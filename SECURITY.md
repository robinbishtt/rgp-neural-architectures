# Security Policy

## Supported Versions

The table below lists which versions of this project currently receive security updates.

| Version | Supported |
|---|---|
| 1.0.x | ✅ Active |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it responsibly. **Do not open a public GitHub issue** for security-related matters.

Report vulnerabilities by opening a [GitHub Security Advisory](https://github.com/robinbishtt/rgp-neural-architectures) through the repository's Security tab. This channel is private and ensures the issue is reviewed before any public disclosure.

Include the following in your report:

- A description of the vulnerability and its potential impact.
- Steps to reproduce the issue or a minimal proof-of-concept.
- The version or commit hash where the vulnerability was observed.
- Any suggested remediation if you have one.

## Response Timeline

Reports will be acknowledged within 72 hours. If the vulnerability is confirmed, a patch will be prepared and released as soon as possible, typically within 14 days for critical issues. You will be notified when the fix is published.

## Scope

This is a research codebase. The primary security considerations are:

**Dependency vulnerabilities.** All Python dependencies are pinned to exact versions in `requirements.txt`. Reviewers should run `pip audit` to check for known CVEs in the pinned versions before using this code in a production or sensitive environment.

**Container images.** The `Dockerfile` and `Singularity.def` files base their images on `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`. Container users are responsible for assessing whether the base image is appropriate for their security requirements.

**Data integrity.** The codebase uses SHA-256 checksums (`src/provenance/`) to verify dataset integrity. These checksums protect against accidental data corruption but are not a security control against adversarial data tampering.

**No network exposure.** This codebase does not expose network services. The Docker Compose configuration exposes Jupyter (port 8888) and TensorBoard (port 6006) on localhost only. These should not be exposed to public networks without additional authentication.

## Out of Scope

Theoretical or mathematical issues with the research methodology are not security vulnerabilities. Please open a regular GitHub Issue or Discussion for those.
 