from .models.videoblip import VideoBlipForClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from typing import Any, Callable, Optional
import logging
import pandas as pd
from transformers import (
    Blip2Processor,
    BatchEncoding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

model = VideoBlipForClassification.from_pretrained_videoblip(
    MODEL_NAME,
    num_classes=NUM_CLASSES,
    freeze_vision=True,
    freeze_qformer=False,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

class VideoBlipTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        
        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=labels,
            return_dict=not return_outputs,
        )

        if return_outputs:
          
            loss = outputs[0]
            return (loss, outputs)
        else:
           
            loss = outputs.loss
            return loss

training_args = TrainingArguments(
    run_name="VideoBlip-Classification",
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    report_to="wandb",
)

trainer = VideoBlipTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


train_result = trainer.train()

# Save the best model
trainer.save_model(f"{OUTPUT_DIR}/best_model")
processor.save_pretrained(f"{OUTPUT_DIR}/best_model")

# Log training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

val_metrics = trainer.evaluate(eval_dataset=val_dataset)
trainer.log_metrics("val", val_metrics)
trainer.save_metrics("val", val_metrics)
print(f"\nValidation Results: {val_metrics}")


test_output = trainer.predict(test_dataset=test_dataset)

test_preds = np.argmax(test_output.predictions, axis=-1)
test_labels = test_output.label_ids

# Overall metrics
test_metrics = test_output.metrics
trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)
print(f"\nTest Results: {test_metrics}")
