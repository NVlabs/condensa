#!/bin/bash

VERBOSE=0
if [[ $1 == "-v" ]] || [[ $1 == "--verbose" ]]; then
  VERBOSE=1
fi

for f in $(find test -name '*.py'); do
  if [[ $VERBOSE -eq 1 ]]; then
    echo "[Condensa Test] $f"
  fi
  python3 $f
done
