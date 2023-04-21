![](./docs/source/_resources/aisee-logo.png)

---

An open-source library for computer vision built on top of PyTorch and Timm libraries. It provides an easy-to-use interface for training and predicting with State-of-the-Art neural networks.

## AISee key features 

- 🤗 Simple interface for training and predicting using timm library models.
- 📁 Easily load images from a folder, a pandas `DataFrame` or a single image path.
- 🏋🏽‍♂️ Train SOTA neural networks from pre-trained weights or from scratch in very few lines of code.  
- 🖼️ Supports multiclass and multilabel image classification tasks.
- 🔨 We take care of `DataLoaders`, image transformations and training and inference loops.


## Quick tour

Here's an example of how to quickly train a model using AISee. We just have to initialize a `VisionClassifier` model and create a `Trainer`. As easy as it gets!

```python
from aisee import Trainer, VisionClassifier

# Initialize a VisionClassifier model
model = VisionClassifier(
    model_name="vgg11_bn", 
    num_classes=4,
)

# Create a Trainer 
trainer = Trainer(
    base_model=model, 
    data=f"animals/",
    output_dir="trained_weights.pt",
)

# Train
trainer.train()
```

To predict call `predict` method, we take care of the rest:

```python
# Predict 
trainer.base_model.predict("animals/without_label")
```


## Installation

Install AISee using pip.

```bash
pip install aisee
```

## Getting started

- Visit [AISee Getting started](https://aisee.readthedocs.io/getting_started/) to get an overview of how AISee works.

- Explore [AISee Documentation](https://aisee.readthedocs.io/) page for a detailed guide and comprehensive API reference.

- Check out [AISee Examples](https://aisee.readthedocs.io/examples/) for Jupyter Notebook examples and tutorials, showcasing how to use AISee effectively in various scenarios.

## Contributing

We value community contributions, and we encourage you to get involved in the continuous development of AISee. Please refer to [AISee Development](https://aisee.readthedocs.io/development) page for guidelines on how to contribute effectively.

We also appreciate your feedback which helps us develop a robust and efficient solution for computer vision tasks. Together, let's make AISee the go-to library for AI practitioners and enthusiasts alike.

## Contact Us
For any questions or inquiries regarding the AISee library or potential collaborations, please feel free to contact us in marketing@iic.uam.es. 

## Instituto de Ingeniería del Conocimiento (IIC)
IIC is a non-profit R&D centre founded in 1989 that has been working on Big Data analysis and Artificial Intelligence for more than 30 years. Its value proposition is the development of algorithms and analytical solutions based on applied research tailored to different businesses in different sectors such as energy, health, insurance and talent analytics.

---
