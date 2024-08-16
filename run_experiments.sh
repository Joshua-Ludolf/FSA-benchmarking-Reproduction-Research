#!/bin/bash

mkdir -p output

echo "Running csr.py for index A"
python3 scripts/csr.py -p data -i index/A_index.csv > ./output/csr_A.out 2>&1

echo "Running csr.py for index B"
python3 scripts/csr.py -p data -i index/B_index.csv > ./output/csr_B.out 2>&1

echo "Running multi_pred.py for index A"
python3 scripts/multi_pred.py -p data -i index/A_index.csv > ./output/multi_pred_A.out 2>&1

echo "Running multi_pred.py for index B"
python3 scripts/multi_pred.py -p data -i index/B_index.csv > ./output/multi_pred_B.out 2>&1

echo "Running lstm.py for cluster A and B"
python3 scripts/lstm.py -p data -i index/all_drive_info.csv -t cluster_A> ./output/lstm.out 2>&1

echo "Running patchTST.py for index A and B"
python3 scripts/patchTST.py -p data -i index/all_drive_info.csv -t cluster_A> ./output/patchTST.out 2>&1

echo "Running GPT-4.py for index A"
python3 scripts/GPT-4.py -p data -i index/A_index.csv > ./output/GPT-4_A.out 2>&1

echo "Running GPT-4.py for index B"
python3 scripts/GPT-4.py -p data -i index/B_index.csv > ./output/GPT-4_B.out 2>&1

echo "done"

echo "Compressing..."
tar -czvf output.tar.gz output
