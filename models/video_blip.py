import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

@dataclass
class VideoClassificationOutput:

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class VideoBlipVisionModel(Blip2VisionModel):

    """Video-aware BLIP2 vision model."""

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    )-> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch, _, time, _, _ = pixel_values.size()
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        seq_len = vision_outputs.last_hidden_state.size(1)

        last_hidden_state = vision_outputs.last_hidden_state.view(
            batch, time * seq_len, -1
        )
        pooler_output = vision_outputs.pooler_output.view(batch, time, -1)

        hidden_states = (
            tuple(
                hidden.view(batch, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )

        attentions = (
            tuple(
                hidden.view(batch, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions
            )
            if vision_outputs.attentions is not None
            else None
        )

        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        
        return (last_hidden_state, pooler_output, hidden_states, attentions)


class VideoBlipForClassification(Blip2ForConditionalGeneration):

    """
    Prompt-conditioned VideoBLIP classifier.

    Inputs:
        - pixel_values: (B, C, T, H, W)
        - input_ids: (B, L)
        - attention_mask: (B, L)

    The prompt is fed into Q-Former along with query tokens and video features.

    """

    def __init__(self, config: Blip2Config, num_classes: int) -> None:
        
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.num_classes = num_classes

        self.vision_model = VideoBlipVisionModel(config.vision_config)  

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        qformer_hidden = config.qformer_config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(qformer_hidden),
            nn.Linear(qformer_hidden, qformer_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(qformer_hidden, num_classes),
        )

        self.language_projection = None
        self.language_model = None

        self.post_init()
    
    @classmethod
    def from_pretrained_videoblip(
        cls,
        pretrained_model_name_or_path: str,
        num_classes: int,
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        **kwargs,
    ) -> "VideoBlipForClassification":
        config = Blip2Config.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, num_classes=num_classes)

        pretrained = Blip2ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        msg = model.vision_model.load_state_dict(
            pretrained.vision_model.state_dict(), strict=False
        )
        print(f"Vision model load info: {msg}")
        model.qformer.load_state_dict(pretrained.qformer.state_dict())
        model.query_tokens.data.copy_(pretrained.query_tokens.data)

        if freeze_vision:
            for param in model.vision_model.parameters():
                param.requires_grad = False
            print("Vision encoder frozen.")
        if freeze_qformer:
            for param in model.qformer.parameters():
                param.requires_grad = False
            model.query_tokens.requires_grad = False
            print("Q-Former frozen.")

        del pretrained
        return model
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True, # This will be passed by the Trainer
    ) -> VideoClassificationOutput:
        
        # The Trainer will set return_dict=False during evaluation.
        # We need to respect that and return a tuple.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        frame_embeds = vision_outputs.last_hidden_state
        frame_attention_mask = torch.ones(
            frame_embeds.size()[:-1], dtype=torch.long, device=frame_embeds.device
        )

        query_tokens = self.query_tokens.expand(frame_embeds.size(0), -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=frame_embeds,
            encoder_attention_mask=frame_attention_mask,
            return_dict=return_dict, # Pass return_dict to the qformer
        )
        query_output = query_outputs[0] if not return_dict else query_outputs.last_hidden_state
        pooled_query_output = query_output.mean(dim=1)
        logits = self.classifier(pooled_query_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        if not return_dict:
            # The trainer expects (loss, logits, ...)
            # During evaluation, loss is computed first, then logits are extracted from the rest of the tuple.
            # So the tuple should be (logits, hidden_states, attentions)
            output = (logits,)
            if output_hidden_states:
                output = output + (query_outputs.hidden_states,)
            if output_attentions:
                output = output + (query_outputs.attentions,)
            return ((loss,) + output) if loss is not None else output

        return VideoClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=query_outputs.hidden_states,
            attentions=query_outputs.attentions,
        )
