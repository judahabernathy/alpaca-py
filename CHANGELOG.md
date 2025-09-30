# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- FastAPI edge service module with structured logging, retry helpers, and a shared HTTP client.
- GitHub Actions matrix CI with linting, tests, and OpenAPI artifacts.
- Windows quickstart instructions and developer tooling extras.

### Changed
- Migrated startup/shutdown logic to FastAPI lifespan hooks and enforced an 85% coverage gate.

