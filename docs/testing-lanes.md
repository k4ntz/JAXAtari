# Testing Lanes

Use these commands to run the same lanes used by CI.

Parallel-safe tests run with xdist, while tests marked `serial` run separately without xdist.

## Smoke (default PR lane)

```bash
pytest tests/
```

Run for one game:

```bash
pytest --game phoenix tests/
```

To run exactly like CI for one game:

```bash
pytest -n 2 --dist=loadfile --game phoenix -m "not serial" tests/
pytest --game phoenix -m "serial" tests/
```

## Core Integration (core/wrapper infrastructure PRs)

Representative games:
- `seaquest`
- `kangaroo`
- `montezumarevenge`
- `pong`
- `phoenix`

Example:

```bash
pytest --game seaquest --integration --wrapper-full tests/
pytest --game kangaroo --integration --wrapper-full tests/
pytest --game montezumarevenge --integration --wrapper-full tests/
pytest --game pong --integration --wrapper-full tests/
pytest --game phoenix --integration --wrapper-full tests/
```

## Full on Master Merge

Run all games and all markers (including `slow`) via the `on-master-merge` workflow.
For local verification on one game:

```bash
pytest --game montezumarevenge --slow tests/
```
