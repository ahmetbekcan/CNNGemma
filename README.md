# CNNGemma

## Title
Vision-Language Model with CNN-Based Image Encoding for Satellite Image Captioning

## Overview

The literature survey highlights a trade-off in current vision-language models (VLMs): while vision transformer-based encoders like SigLIP offer rich image representations, they are computationally expensive and slow. On the other hand, models that discard image encoding entirely—such as Fuyu-style architectures—sacrifice training efficiency and generalization by losing reusable visual representations.

This project proposes an alternative image encoding strategy aimed at balancing computational efficiency with training effectiveness. Specifically, I propose replacing the vision transformer encoder in PaliGemma with a lightweight CNN-based encoder. This encoder will transform raw image pixels into intermediate representations suitable for input into the language model decoder. CNNs are well-known for their computational efficiency and inductive biases suited for spatial data, making them a promising alternative to transformer-based vision encoders, especially in domain-specific tasks like satellite image captioning.

## Methodology

### Encoder Design:
A lightweight convolutional neural network (CNN) model will be used to encode satellite images into feature representations. This model will replace the SigLIP model in the original PaliGemma architecture.

### Integration with Language Model:
The CNN outputs will be projected into a format compatible with the decoder-only language model in PaliGemma.

### Training and Evaluation:
The CNN model will be trained on the RISC dataset using existing caption annotations. The language model will be frozen to increase the training speed. Performance will be evaluated using captioning metrics such as BLEU, METEOR, and CIDEr, as well as computational metrics like inference speed and memory usage.

### Comparative Analysis:
Due to limited resources, it is not possible to compare SigLIP with CNN based encoders. Instead, different CNN encoders (MobileNetV3 Large and EfficientNet B0) that are pretrained with ImageNet data will be fine-tuned on RISCM dataset, and their performances will be compared to each other. Also, the performances of different tokenization techniques will be compared. The first tokenization technique is taking the final output of CNN encoders as a single image token, and the second one is extracting the feature map of the CNN models as multiple tokens.

## Novelty and Conceptual Merits

By introducing a middle ground between full transformer encoders and no encoders, this project explores a novel architecture that leverages the efficiency of CNNs without compromising training dynamics. Developing a lightweight model for the classification of satellite images can be beneficial in real-world scenarios with limited computational resources. Finally, this approach offers insights into how hybrid architectures can enhance the efficiency and adaptability of vision-language models for specialized domains.
