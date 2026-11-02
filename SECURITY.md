# Security policy

## Reporting a vulnerability

Do not open a public issue. Use **Security > Report a vulnerability** on this repository
(GitHub private vulnerability reporting), or email the product's security team. Include what you
found, how to reproduce it, and the impact you expect. We acknowledge reports within two
business days.

## What this repository does by default

- Secret scanning with push protection; Dependabot alerts and security updates.
- CodeQL code scanning on every pull request; merges are blocked on new high or critical alerts.
- Dependency review on pull requests; vulnerable additions at the platform's threshold block.
- The default branch is protected: pull requests only, code-owner review, required checks.

## Handling sensitive data

This repository may belong to a product that handles PHI or payment data. Never commit real
patient or cardholder data, including in tests, fixtures, logs or screenshots. Use synthetic
data. Do not log request bodies, identifiers or tokens.
