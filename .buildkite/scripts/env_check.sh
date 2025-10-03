#!/bin/bash

if [[ ${#GITHUB_PAT} -gt 0 ]]; then
  echo "Successful read secret key"
else
  echo "Unable to read secret key"
  exit 1
fi
