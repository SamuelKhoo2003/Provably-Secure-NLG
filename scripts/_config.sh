#!/usr/bin/env bash

config_value() {
  local key="$1"
  local default_value="${2:-}"
  local config_path="${CONFIG:-}"

  if [[ -z "$config_path" || ! -f "$config_path" ]]; then
    printf '%s\n' "$default_value"
    return 0
  fi

  local value
  value="$(
    awk -F: -v key="$key" '
      $1 == key {
        sub(/^[[:space:]]*/, "", $2)
        sub(/[[:space:]]*$/, "", $2)
        gsub(/[\047"]/, "", $2)
        gsub(/^\[/, "", $2)
        gsub(/\]$/, "", $2)
        gsub(/[[:space:]]+/, "", $2)
        print $2
        exit
      }
    ' "$config_path"
  )"

  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$default_value"
  fi
}

config_first_value() {
  local value
  value="$(config_value "$1" "${2:-}")"
  printf '%s\n' "${value%%,*}"
}
