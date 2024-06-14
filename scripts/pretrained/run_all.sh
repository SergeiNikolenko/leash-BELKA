#!/bin/bash

cd "$(dirname "$0")"

# Разбивка данных на чанки
python split_data.py

# Подбор оптимального размера батча
python estimate_batch_size.py

# Обработка каждого чанка
for chunk_file in ../../data_chunks/chunk_*.parquet
do
    echo "Processing $chunk_file"
    python process_chunk.py $chunk_file
done

# Объединение всех обработанных чанков в один файл
python combine_chunks.py
