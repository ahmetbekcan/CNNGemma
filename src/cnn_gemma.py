import torch.nn as nn
from enum import Enum
import torchvision.models as models
import torch 
from gemma import GemmaForCausalLM, KVCache, GemmaConfig
from typing import Optional, Tuple

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
        self.features = self.model.features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if (self.config.token_type == CNNTokenType.Single):
            x = self.model.avgpool(x)
            x = x.flatten(1)
            x = x.unsqueeze(1)
            print(f"Mobile single token: {x.shape}")
            return x
        b,c,h,w = x.shape
        x = x.view(b,c,h*w)
        x = x.permute(0,2,1)
        print(f"Mobile multiple token: {x.shape}")
        return x
    
class EfficientNetImageEncoder(nn.Module):
    def __init__(self, config: CNNImageEncoderConfig):
        super().__init__()
        assert(config.architecture == CNNArchitecture.EfficientNetB0)
        self.config = config
        self.model = models.efficientnet_b0(pretrained=True)
        self.features = self.model.features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if (self.config.token_type == CNNTokenType.Single):
            x = self.model.avgpool(x)
            x = x.flatten(1)
            x = x.unsqueeze(1)
            print(f"Efficient single token: {x.shape}")
            return x
        b,c,h,w = x.shape
        x = x.view(b,c,h*w)
        x = x.permute(0,2,1)
        print(f"Efficient multiple token: {x.shape}")
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

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
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
