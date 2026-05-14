#!/usr/bin/env bash

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PYTHON_BIN" ]]; then
      printf '%s\n' "$PYTHON_BIN"
      return 0
    fi

    echo "Python executable not found at $PYTHON_BIN" >&2
    return 1
  fi

  if [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "Python executable not found. Set PYTHON_BIN or create .venv first." >&2
  return 1
}