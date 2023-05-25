[![AISee Logo](./docs/source/_resources/aisee-logo.png)](https://iiconocimiento.github.io/aisee/stable/)

---

[![License: MIT](https://img.shields.io/github/license/iiconocimiento/aisee)](https://github.com/iiconocimiento/aisee/blob/main/LICENSE) ![GitHub Stars](https://img.shields.io/github/stars/iiconocimiento/aisee?style=social) [![Latest Version on Pypi](https://img.shields.io/pypi/v/aisee)](https://pypi.org/project/aisee/) ![Supported Python versions](https://img.shields.io/pypi/pyversions/aisee) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Unit tests](https://github.com/iiconocimiento/aisee/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/github/iiconocimiento/aisee/branch/main/graph/badge.svg?token=YA7QJ8FOIM)](https://codecov.io/github/iiconocimiento/aisee) [![Docs](https://github.com/iiconocimiento/aisee/actions/workflows/documentation.yml/badge.svg)](https://iiconocimiento.github.io/aisee/stable/)

An open-source library for computer vision built on top of PyTorch and Timm libraries. It provides an easy-to-use interface for training and predicting with State-of-the-Art neural networks.

## AISee key features 

- 🤗 Simple interface for training and predicting using timm library models.
- 📁 Easily load images from a folder, a pandas `DataFrame` or a single image path.
- 🏋🏽‍♂️ Train SOTA neural networks from pre-trained weights or from scratch in very few lines of code.  
- 🖼️ Supports multiclass and multilabel image classification tasks.
- 🔨 We take care of `DataLoaders`, image transformations and training and inference loops.


## Installation

Install AISee using pip.

```bash
pip install aisee
```


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


## Try it on Google Colab
<a target="_blank" href="https://colab.research.google.com/github/iiconocimiento/aisee/blob/main/notebooks/multi_class_classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Getting started

- Visit [AISee Getting started](https://iiconocimiento.github.io/aisee/stable/_getting_started/getting_started.html) to get an overview of how AISee works.

- Explore [AISee Documentation](https://iiconocimiento.github.io/aisee/stable/) page for a detailed guide and comprehensive API reference.

- Check out [AISee Examples](https://iiconocimiento.github.io/aisee/stable/_examples/examples.html) for Jupyter Notebook examples and tutorials, showcasing how to use AISee effectively in various scenarios.

## Contributing

We value community contributions, and we encourage you to get involved in the continuous development of AISee. Please refer to [AISee Development](https://iiconocimiento.github.io/aisee/stable/_development/development.html) page for guidelines on how to contribute effectively.

We also appreciate your feedback which helps us develop a robust and efficient solution for computer vision tasks. Together, let's make AISee the go-to library for AI practitioners and enthusiasts alike.

## Contact Us
For any questions or inquiries regarding the AISee library or potential collaborations, please feel free to contact us in marketing@iic.uam.es. 

## Instituto de Ingeniería del Conocimiento (IIC)
[IIC](https://www.iic.uam.es/) is a non-profit R&D centre founded in 1989 that has been working on Big Data analysis and Artificial Intelligence for more than 30 years. Its value proposition is the development of algorithms and analytical solutions based on applied research tailored to different businesses in different sectors such as energy, health, insurance and talent analytics.

---
