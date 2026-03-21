# Contributing to CLARA

## Development Setup

```bash
git clone https://github.com/yoonsik-jung-opt/clara-opt.git
cd clara-opt
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

- Python 3.10+, type hints required on all public functions
- Format: `ruff format clara/ tests/`
- Lint: `ruff check clara/ tests/`
- Tests: `pytest tests/`

## Branch Convention

- `feat/*` — new features
- `fix/*` — bug fixes
- `bench/*` — benchmarks
- `paper/*` — paper writing
- All PRs target `dev` branch

## Commit Convention

`feat:`, `fix:`, `test:`, `bench:`, `paper:`, `docs:`, `refactor:`, `ci:`, `chore:`

## Testing

Every new feature must include tests. Run the full suite before submitting:

```bash
pytest tests/ -v
```

Cross-validation tests compare Internal Simplex vs HiGHS — both engines
must agree on optimal values, duals, and sensitivity ranges.
