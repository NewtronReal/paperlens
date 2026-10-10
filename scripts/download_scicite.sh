#!/usr/bin/env bash
# Fallback download of SciCite (Cohan et al., NAACL 2019) if the Hugging Face hub copy does not load.
# Produces data/scicite/{train,dev,test}.jsonl
set -euo pipefail
mkdir -p data
cd data
curl -L -o scicite.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
tar xzf scicite.tar.gz
ls -la scicite
