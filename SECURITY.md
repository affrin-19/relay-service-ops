# Security Policy

## Supported Versions

We actively maintain and provide security patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The Relay team takes the security of automated service management workflows seriously. If you discover a security vulnerability, credential exposure, or authorization bypass:

1. **Do not create a public GitHub Issue.**
2. Email your findings confidentially to the maintainers at `security@relay-ops.internal` (or directly via the repository owner's GitHub profile contact).
3. Include the following details:
   - Severity and scope of the vulnerability.
   - Proof of concept (PoC) or reproducible steps.
   - Potential impact on tenant isolation, LLM prompt injection, or database records.

### Response Timeline
- **Initial Response:** Within 48 hours of receipt.
- **Triage & Assessment:** Within 5 business days.
- **Patch Release:** Target within 14 business days depending on complexity.

### Safe Harbor & Responsible Disclosure
We commit to not pursuing legal action against security researchers who conduct testing in good faith, avoid privacy violations, do not destroy data, and provide reasonable time to remediate before public disclosure.
