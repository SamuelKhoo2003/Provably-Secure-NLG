#!/usr/bin/env bash
# Helper to configure Gurobi environment variables for this user/project.
# Usage: source scripts/env.sh

# If you already know the install dir, export GUROBI_INSTALL_DIR before sourcing.
GUROBI_INSTALL_DIR="${GUROBI_INSTALL_DIR:-}"

find_candidate() {
  # look for likely install locations in order
  local dirs=("$HOME/opt" "$HOME/.local" "/opt" "$HOME")
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
      # prefer explicit linux64 subdir if present
      for match in "$d"/gurobi*; do
        [[ -e "$match" ]] || continue
        echo "$match"
        return 0
      done
    fi
  done
  return 1
}

if [[ -z "$GUROBI_INSTALL_DIR" ]]; then
  GUROBI_INSTALL_DIR="$(find_candidate || true)"
fi

if [[ -z "$GUROBI_INSTALL_DIR" ]]; then
  echo "No Gurobi installation found. Set GUROBI_INSTALL_DIR to the install path and source this file again." >&2
  return 1 2>/dev/null || exit 1
fi

# Many Gurobi tarballs unpack to a top-level folder that contains a linux64/ subdir.
if [[ -d "$GUROBI_INSTALL_DIR/linux64" ]]; then
  export GUROBI_HOME="$GUROBI_INSTALL_DIR/linux64"
else
  export GUROBI_HOME="$GUROBI_INSTALL_DIR"
fi

export PATH="$GUROBI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"

echo "Configured GUROBI_HOME=$GUROBI_HOME"

# Optional: activate project venv if present
if [[ -f .venv/bin/activate ]]; then
  echo "To also activate the project's virtualenv, run: source .venv/bin/activate"
fi

echo "You can now run: grbgetkey <YOUR-LICENSE-KEY>"
