# Transformer with Key-Value Compression

This repository contains a PyTorch implementation of the Transformer model, modified to include Key-Value (KV) compression in the Multi-Head Attention mechanism.

## Paper Summary

The Transformer model ("Attention Is All You Need") uses a self-attention mechanism to process sequences. The standard Multi-Head Attention scales quadratically with sequence length in terms of both time and memory complexity ($O(N^2 \cdot d_{model})$).

## Code Description

This implementation modifies the standard Transformer architecture to improve efficiency:

-   **KV Compression**: The Key and Value matrices in the Multi-Head Attention module are projected to a compressed dimension $k$ (set to 128) instead of the full model dimension $d_{model}$ (typically 512).
-   **Complexity Reduction**: This reduces the complexity of the attention computation from $O(N \cdot d_{model})$ to $O(N \cdot k)$, resulting in a ~4x speedup and memory reduction in the attention layer.
-   **Structure**: The codebase is organized into modules (`transformer/`) for the model components and scripts (`train.py`, `translate.py`, `preprocess.py`) for the training and inference workflow.

## Usage

1.  **Preprocess Data**:
    ```bash
    python preprocess.py -raw_dir raw_data -data_dir processed_data -save_data data.pkl -codes codes.txt -prefix m30k
    ```

2.  **Train Model**:
    ```bash
    python train.py -data_pkl processed_data/data.pkl -output_dir output -use_tb
    ```

3.  **Translate**:
    ```bash
    python translate.py -model output/model.chkpt -data_pkl processed_data/data.pkl -output prediction.txt
    ```
