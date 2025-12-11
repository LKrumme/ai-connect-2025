---
dataset_info:
- config_name: grid_mode
  features:
  - name: id
    dtype: string
  - name: size
    dtype: string
  - name: puzzle
    dtype: string
  - name: solution
    struct:
    - name: header
      sequence: string
    - name: rows
      sequence:
        sequence: string
  - name: created_at
    dtype: string
  splits:
  - name: test
    num_bytes: 1545275
    num_examples: 1000
  download_size: 345826
  dataset_size: 1545275
- config_name: mc_mode
  features:
  - name: id
    dtype: string
  - name: puzzle
    dtype: string
  - name: question
    dtype: string
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: created_at
    dtype: string
  splits:
  - name: test
    num_bytes: 5039993
    num_examples: 3259
  download_size: 826292
  dataset_size: 5039993
configs:
- config_name: grid_mode
  data_files:
  - split: test
    path: grid_mode/test-*
- config_name: mc_mode
  data_files:
  - split: test
    path: mc_mode/test-*
---
