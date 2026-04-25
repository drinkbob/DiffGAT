# Contributing to DiffGAT

Thanks for your interest in contributing.

This document explains how to propose changes, report issues, and keep contributions consistent with the project workflow.

## 1. Ways to Contribute

- Report bugs
- Propose new features
- Improve documentation
- Add tests, validation scripts, or reproducibility utilities
- Improve model components and evaluation scripts

## 2. Before You Start

1. Check existing issues and pull requests to avoid duplicate work.
2. Open an issue first for large changes (new architecture, major refactor, metric definition changes).
3. Keep scope focused: one pull request should solve one problem.

## 3. Development Setup

Use Python 3.9+.

Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

Run the preflight check:

```bash
py -3 repro_check.py --train_csv benzimidazole_dataset.csv
```

## 4. Branching and Commits

- Create a feature branch from your default branch.
- Use clear commit messages (imperative style), for example:
  - `fix baseline CSV parsing for missing headers`
  - `add deterministic seed control for multi-core generator`
- Keep commit history clean and relevant.

## 5. Pull Request Checklist

Before opening a PR, make sure:

- [ ] The change is documented in code comments or docs where needed.
- [ ] Relevant scripts run successfully.
- [ ] No generated result artifacts are intentionally committed.
- [ ] README or docs are updated if usage behavior changed.
- [ ] The PR description explains:
  - what changed,
  - why it changed,
  - how it was validated.

## 6. Coding Guidelines

- Prefer small, composable functions.
- Avoid hard-coded machine-specific paths.
- Keep CLI parameters explicit and reproducible.
- Preserve backward compatibility unless a breaking change is justified and documented.
- For stochastic logic, expose and respect seed controls.

## 7. Reproducibility Expectations

For changes affecting experiments or metrics:

- Specify random seed behavior.
- Document input/output files.
- Include the exact command used for validation.
- Keep metric definitions consistent with existing scripts.

## 8. Issue Reporting Template

When reporting bugs, include:

- OS and Python version
- Command used
- Full error message/traceback
- Minimal input required to reproduce
- Expected vs actual behavior

## 9. Security and Sensitive Data

- Do not commit credentials, tokens, or personal data.
- Avoid committing large generated artifacts unless explicitly required.

## 10. License

By contributing, you agree that your contributions are licensed under the repository's MIT License.
