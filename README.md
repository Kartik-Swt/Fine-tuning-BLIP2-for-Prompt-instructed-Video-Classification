# Fine-tuning BLIP-2 for Prompt-Instructed Video Classification

Fine-tune [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2) for video classification where a **natural-language prompt** conditions the Q-Former, replacing the language model head with a lightweight classification head. The vision encoder is kept frozen while the Q-Former and classifier are trained end-to-end.

---

## How It Works

```
Video frames (T, H, W, C)
        │
        ▼
VideoBlipVisionModel          ← frozen BLIP-2 ViT, processes all frames jointly
        │   (B, T×seq_len, D)
        ▼
Blip2QFormerModel             ← trained; query tokens cross-attend to frame features
        │   (B, num_queries, D)
        ▼
Mean pooling → MLP classifier → class logits
```

A text **prompt** (e.g. `"Classify the activity in this video."`) is tokenised and fed into the Q-Former together with the learnable query tokens, guiding the model toward task-relevant visual features.

---

## Repository Structure

```
.
├── models/
│   └── video_blip.py       # VideoBlipVisionModel & VideoBlipForClassification
├── train.py                # Training loop using HuggingFace Trainer
└── utils.py                # Video I/O, preprocessing, dataset helpers, metrics
```

### Key Components

| File | Class / Function | Purpose |
|------|-----------------|---------|
| `models/video_blip.py` | `VideoBlipVisionModel` | Extends `Blip2VisionModel` to accept `(B, C, T, H, W)` video tensors by flattening the temporal dimension before the ViT and reshaping outputs back |
| `models/video_blip.py` | `VideoBlipForClassification` | Extends `Blip2ForConditionalGeneration`; drops the language model, adds a `LayerNorm → Linear → GELU → Dropout → Linear` classifier on top of the pooled Q-Former output |
| `utils.py` | `read_video_pyav` | Reads a video file with PyAV and uniformly samples `NUM_FRAMES` frames |
| `utils.py` | `process` | Runs the BLIP-2 processor on frames + prompt and reshapes `pixel_values` to `(C, T, H, W)` |
| `utils.py` | `create_dataset` | Maps `process_example` over a HuggingFace `DatasetDict` in parallel |
| `utils.py` | `collate_fn` | Batches `pixel_values` and `labels` tensors for the Trainer |
| `utils.py` | `compute_metrics` | Returns **accuracy** and **weighted F1** during evaluation |
| `train.py` | `VideoBlipTrainer` | Subclasses `Trainer` to implement a custom `compute_loss` that unpacks `pixel_values` and `labels` |

---

## Requirements

```
torch
transformers
datasets
scikit-learn
pandas
av          # PyAV – video decoding
wandb       # experiment tracking
```

Install with:

```bash
pip install torch transformers datasets scikit-learn pandas av wandb
```

---

## Configuration

Before running, set the following constants (currently referenced as bare names in the scripts):

| Constant | Description |
|----------|-------------|
| `MODEL_NAME` | HuggingFace model ID, e.g. `"Salesforce/blip2-opt-2.7b"` |
| `NUM_CLASSES` | Number of target classes |
| `NUM_FRAMES` | Frames to sample per video (default suggested: `8`) |
| `PROMPT` | Text prompt fed to Q-Former, e.g. `"Classify the activity in this video."` |
| `OUTPUT_DIR` | Directory to save checkpoints and the best model |
| `NUM_EPOCHS` | Training epochs |
| `BATCH_SIZE` | Per-device batch size |
| `LEARNING_RATE` | Optimizer learning rate |

---

## Training

```bash
python train.py
```

The script will:

1. Load `VideoBlipForClassification` from a pretrained BLIP-2 checkpoint, copying vision encoder and Q-Former weights.
2. Freeze the vision encoder; keep the Q-Former and classifier trainable.
3. Train with the HuggingFace `Trainer` using:
   - Gradient accumulation (8 steps)
   - Mixed precision (FP16 on CUDA)
   - Epoch-level evaluation and checkpointing
   - Early stopping (patience = 3 epochs)
   - Best model selection by weighted F1
4. Save the best model and processor to `{OUTPUT_DIR}/best_model`.
5. Log training, validation, and test metrics (also reported to **Weights & Biases**).

### Trainable Parameters

With the vision encoder frozen, only the Q-Former and the classification head are updated — typically **~10–15 % of total parameters**, making fine-tuning feasible on a single GPU.

---

## Model Architecture Details

### `VideoBlipVisionModel`

Processes all video frames simultaneously by:

1. Permuting `(B, C, T, H, W)` → `(B×T, C, H, W)`.
2. Running the standard BLIP-2 ViT.
3. Reshaping `last_hidden_state` back to `(B, T×seq_len, D)` so the Q-Former can cross-attend over all frame tokens.

### `VideoBlipForClassification`

- **Removes** the language projection and language model from `Blip2ForConditionalGeneration`.
- **Adds** a 2-layer MLP classifier on the mean-pooled Q-Former query output.
- Provides `from_pretrained_videoblip()` — a convenience class method that loads pretrained vision and Q-Former weights, then applies optional freezing.

---

## Output

After training the following are produced under `OUTPUT_DIR`:

```
OUTPUT_DIR/
├── best_model/            # Model weights + processor
├── train_results.json
├── val_results.json
└── test_results.json
```

Metrics logged: **accuracy**, **weighted F1**, cross-entropy loss.
README.md

copilot
