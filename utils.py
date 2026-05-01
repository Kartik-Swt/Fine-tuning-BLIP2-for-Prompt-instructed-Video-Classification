import os
import av
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from transformers import (
    Blip2Processor,
    BatchEncoding
)


def read_video_pyav(filepath: str, num_frames: int = 8) -> np.ndarray:

    """
    Read a video file and uniformly sample `num_frames` frames.

    :returns: np.ndarray of shape (num_frames, height, width, 3), dtype uint8

    """
    container = av.open(filepath)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:

        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return np.stack([frames[i] for i in indices])

    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break

    container.close()

    while len(frames) < num_frames:
        frames.append(frames[-1])  # repeat last frame

    return np.stack(frames)

def process(
        processor: Blip2Processor,
        video: np.ndarray | None = None, # Expects numpy array
        text: str | list[str] | None = None,
)-> BatchEncoding:
    
    if video is not None:

        video_list = [frame for frame in video]
        
    inputs = processor(images=video_list, text=text, return_tensors="pt")

    if video is not None:

        if isinstance(inputs['pixel_values'], list):
            pixel_values = torch.stack(inputs['pixel_values'])
        else:
            pixel_values = inputs['pixel_values']


        num_frames = video.shape[0]
        c, h, w = pixel_values.shape[1:]
        inputs["pixel_values"] = pixel_values.view(num_frames, c, h, w).permute(1, 0, 2, 3)
    
    return inputs

def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])  # (B, C, T, H, W)
    labels = torch.tensor([item['labels'] for item in batch])  # (B,)
    return {'pixel_values': pixel_values, 'labels': labels}


def create_dataset(data: DatasetDict, processor: Blip2Processor) -> DatasetDict:
    
    def process_example(example: dict) -> dict:
        video_path = example['video']
        label = example['label']
        video_frames = read_video_pyav(video_path, num_frames=NUM_FRAMES) # (T, H, W, C) numpy array
        # Pass the numpy array directly
        inputs = process(processor, video=video_frames, text=PROMPT)
        return {
            'pixel_values': inputs['pixel_values'],  # (C, T, H, W)
            'labels': label
        }
    num_cpus = os.cpu_count()
    return data.map(process_example, batched=False, num_proc=num_cpus)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred_ids),
        "f1": f1_score(labels, pred_ids, average="weighted"),
    }
