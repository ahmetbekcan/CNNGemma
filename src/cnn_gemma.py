import torch.nn as nn
from enum import Enum
import torchvision.models as models
import torch 
from gemma import GemmaForCausalLM, KVCache, GemmaConfig
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput

class CNNArchitecture(Enum):
    EfficientNetB0 = "EfficientNetB0"
    MobileNetV3_Large = "MobileNetV3_Large"

class CNNTokenType(Enum):
    Multiple = "Multiple"
    Single = "Single"

class CNNImageEncoderConfig():
    def __init__(self,
            architecture: CNNArchitecture,
            token_type: CNNTokenType,
            hidden_size=2048,
            image_size=224,
            **kwargs):
        self.architecture = architecture
        self.token_type = token_type
        self.hidden_size = hidden_size
        self.image_size = image_size

        if (self.architecture == CNNArchitecture.MobileNetV3_Large):
            self.image_token_size = 960
        elif (self.architecture == CNNArchitecture.EfficientNetB0):
            self.image_token_size = 1280
        else:
            raise ValueError("This model is not implemented!")
        
        if (self.token_type == CNNTokenType.Single):
            self.num_image_tokens = 1
        else:
            self.num_image_tokens = 49

    def to_dict(self):
        return {
            "architecture": self.architecture.name if hasattr(self.architecture, "name") else self.architecture,
            "token_type": self.token_type.name if hasattr(self.token_type, "name") else self.token_type,
            "hidden_size": self.hidden_size,
            "image_size": self.image_size,
            "image_token_size": self.image_token_size,
            "num_image_tokens": self.num_image_tokens,
        }

class CNNImageEncoder(nn.Module):
    def __init__(self, config: CNNImageEncoderConfig):
        super().__init__()
        self.config = config
        if (config.architecture == CNNArchitecture.MobileNetV3_Large):
            self.model = MobileNetImageEncoder(config)
        elif (config.architecture == CNNArchitecture.EfficientNetB0):
            self.model = EfficientNetImageEncoder(config)
        else:
            raise ValueError(f"The model {config.architecture} is not implemented!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

class MobileNetImageEncoder(nn.Module):
    def __init__(self, config: CNNImageEncoderConfig):
        super().__init__()
        assert(config.architecture == CNNArchitecture.MobileNetV3_Large)
        self.config = config
        self.model = models.mobilenet_v3_large(pretrained=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        if (self.config.token_type == CNNTokenType.Single):
            x = self.model.avgpool(x)
            x = x.flatten(1)
            x = x.unsqueeze(1)
            return x
        b,c,h,w = x.shape
        x = x.view(b,c,h*w)
        x = x.permute(0,2,1)
        return x
    
class EfficientNetImageEncoder(nn.Module):
    def __init__(self, config: CNNImageEncoderConfig):
        super().__init__()
        assert(config.architecture == CNNArchitecture.EfficientNetB0)
        self.config = config
        self.model = models.efficientnet_b0(pretrained=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        if (self.config.token_type == CNNTokenType.Single):
            x = self.model.avgpool(x)
            x = x.flatten(1)
            x = x.unsqueeze(1)
            return x
        b,c,h,w = x.shape
        x = x.view(b,c,h*w)
        x = x.permute(0,2,1)
        return x

class CNNProjector(nn.Module):

    def __init__(self, config: CNNImageEncoderConfig):
        super().__init__()
        self.config = config
        self.projection_layer = nn.Linear(self.config.image_token_size, self.config.projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection_layer(x)
        return x

class CNNGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        if vision_config is None:
            vision_config = {}
        else:
            vision_config["architecture"] = CNNArchitecture[vision_config["architecture"]]
            vision_config["token_type"] = CNNTokenType[vision_config["token_type"]]
            
        if text_config is None:
            text_config = {}

        self.vision_config = CNNImageEncoderConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        if (self.vision_config.token_type == CNNTokenType.Single):
            self.text_config.num_image_tokens = 1
        else:
            self.text_config.num_image_tokens = 49

        self.vision_config.projection_dim = projection_dim
    
    def to_dict(self):
        return {
            "ignore_index": self.ignore_index,
            "image_token_index": self.image_token_index,
            "vocab_size": self.vocab_size,
            "projection_dim": self.projection_dim,
            "hidden_size": self.hidden_size,
            "is_encoder_decoder": self.is_encoder_decoder,
            "pad_token_id": self.pad_token_id,
            "vision_config": self.vision_config.to_dict() if hasattr(self.vision_config, "to_dict") else self.vision_config,
            "text_config": self.text_config.to_dict() if hasattr(self.text_config, "to_dict") else self.text_config,
        }
    
class CNNGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: CNNGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = CNNImageEncoder(config.vision_config)
        self.multi_modal_projector = CNNProjector(config.vision_config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.language_model.lm_head.weight.data_ptr() == self.language_model.model.embed_tokens.weight.data_ptr():
            del state['language_model.lm_head.weight']
        return state
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
    
    #copied from hugging face implementation (PaliGemmaForConditionalGeneration)
    def _update_causal_mask(self, attention_mask, token_type_ids, inputs_embeds, kv_cache: KVCache):
        dtype = inputs_embeds.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = inputs_embeds.shape[1]
        past_seen_tokens = kv_cache.num_items() if kv_cache is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        if kv_cache is not None:
            target_length = kv_cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
            )
        return causal_mask
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, ModelOutput]:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)
        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, causal_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        #copied from hugging face implementation (PaliGemmaForConditionalGeneration)
        if labels is not None:
            if self.pad_token_id in labels:
                labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)
            causal_mask = self._update_causal_mask(attention_mask, token_type_ids, inputs_embeds, kv_cache)

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        #copied from hugging face implementation (PaliGemmaForConditionalGeneration)
        logits = outputs["logits"]
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
            outputs["loss"] = loss
        
        return outputs