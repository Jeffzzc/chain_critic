---
dataset_info:
  features:
  - name: orig_instruction
    dtype: string
  - name: orig_score3_description
    dtype: string
  - name: orig_score4_description
    dtype: string
  - name: output
    dtype: string
  - name: orig_response
    dtype: string
  - name: orig_reference_answer
    dtype: string
  - name: orig_feedback
    dtype: string
  - name: orig_score1_description
    dtype: string
  - name: orig_score
    dtype: string
  - name: orig_criteria
    dtype: string
  - name: orig_score2_description
    dtype: string
  - name: instruction
    dtype: string
  - name: orig_score5_description
    dtype: string
  - name: input
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 15401684
    num_examples: 1000
  download_size: 0
  dataset_size: 15401684
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Dataset Card for "Promixtheus-Absolute-Bench"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)