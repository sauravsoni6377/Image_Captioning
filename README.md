# Image Captioning Project

## Overview
This project implements an image captioning model that generates textual descriptions for input images. The model leverages OpenAI's CLIP (Contrastive Language-Image Pre-training) for image feature extraction and GPT-2 for text generation. The framework provides a flexible and scalable approach to train and evaluate captioning models.

## Features
- **CLIP Integration**: Uses CLIP to encode image features.
- **GPT-2 for Text Generation**: Leverages GPT-2 for generating meaningful captions.
- **Customizable Architecture**: Supports Multi-Layer Perceptron (MLP) for feature projection.
- **Beam Search and Nucleus Sampling**: Provides advanced decoding strategies for diverse caption generation.
- **Device Compatibility**: Automatically selects CPU or GPU for computation.

## Requirements
### Libraries
- Python 3.7+
- PyTorch
- Transformers
- NumPy
- scikit-image
- PIL
- tqdm

Install the required libraries using:
```bash
pip install torch transformers numpy scikit-image pillow tqdm
```

## Directory Structure
```
.
├── main.py               # Main script
├── pretrained_models/    # Directory for saving pre-trained weights
└── README.md             # Project documentation
```

## How to Use

### 1. Clone the Repository
```bash
git clone <repository-url>
cd image-captioning-project
```

### 2. Pretrained Models
Download the pre-trained weights and place them in the `pretrained_models/` directory.

### 3. Train the Model
Modify the script as needed to load your dataset and train the model.
```python
# Example: Initialize the model
model = ClipCaptionModel(prefix_length=10)
```

### 4. Generate Captions
Use `generate_beam` or `generate2` to create captions for your images.
```python
captions = generate_beam(
    model, tokenizer, beam_size=5, prompt=None, embed=image_embedding
)
print(captions)
```

### 5. Save and Load Models
Save trained weights:
```python
torch.save(model.state_dict(), 'pretrained_models/model_weights.pt')
```
Load weights:
```python
model.load_state_dict(torch.load('pretrained_models/model_weights.pt'))
```

## Implementation Details
### Classes
- **`MLP`**: Implements a Multi-Layer Perceptron for projecting CLIP features.
- **`ClipCaptionModel`**: Core class combining CLIP embeddings with GPT-2.
- **`ClipCaptionPrefix`**: A variant of `ClipCaptionModel` that freezes GPT-2 weights.

### Text Generation Methods
- **`generate_beam`**: Uses beam search for diverse and high-quality text generation.
- **`generate2`**: Implements top-p nucleus sampling for creative outputs.

## Example Workflow
1. Encode an image using CLIP to extract features.
2. Pass the encoded features to the model.
3. Generate a caption using one of the decoding strategies.

## Customization
- Modify the MLP architecture to adapt the model for specific datasets.
- Experiment with prefix length (`prefix_length`) for balancing memory usage and performance.
- Tune decoding parameters (e.g., `beam_size`, `top_p`) to achieve desired output diversity.

## Future Work
- Support for additional feature extractors (e.g., ViT, ResNet).
- Incorporation of fine-tuning strategies for specific domains.
- Development of an interactive demo for real-time caption generation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- OpenAI for CLIP and GPT-2 models.
- Hugging Face for the Transformers library.

