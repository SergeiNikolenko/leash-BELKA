#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p ../train_preprocessed

# Create a directory for log files if it doesn't exist
mkdir -p ../chunk_logs

# Iterate over all chunk scripts and run them in parallel
for i in {1..10}
do
   python ../chunk_scripts/process_chunk_${i}.py > ../chunk_logs/process_chunk_${i}.log 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All chunk processing scripts have completed."
