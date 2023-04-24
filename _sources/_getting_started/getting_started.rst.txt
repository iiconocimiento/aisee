.. _getting_started:

Getting Started
===============

Welcome to the Getting Started section of our Python library for image classification.
In this section, we will guide you through the first steps to using the library, from
installation to image prediction.


Installation
------------

Before starting to use the library, you must install it. You can do this using pip,
the Python package manager. Open a terminal and type the following command:

.. highlight:: console
.. code-block:: console

    pip install aisee

Making predictions without training a model
-------------------------------------------

In some cases, you may want to use a pre-trained neural network instead of training a
network from scratch. A pre-trained network has already been trained on a massive dataset
and can be used to make predictions on new images.

You can load any model from the timm library:

.. highlight:: python
.. code-block:: python

    import timm

    timm.list_models(pretrained=True)

You can search for models of a specific architecture:

.. highlight:: python
.. code-block:: python

    import timm

    timm.list_models("*vgg11*", pretrained=True)

Once the model has been chosen, it is necessary to instantiate the VisionClassifier.

.. highlight:: python
.. code-block:: python
    
    from aisee import VisionClassifier

    vc = VisionClassifier(
        model_name="vgg11_bn",
        num_classes=4,
        learning_method="freezed",
        task="single_label",
    )

To make the predictions with our library, the data could be in different formats:
as a directory to a single image, as a directory of images or as a dataframe.

Here's an example using the directory to a single image:

.. highlight:: python
.. code-block:: python

    vc.predict("animals/tiger/tiger1.jpg")

Here's an example using a directory of images like this:

.. code-block:: console

            └── animals
              └── cat
                 ├── cat1.jpg
                 ├── cat2.jpg
                 └── ...

.. highlight:: python
.. code-block:: python

    vc.predict("animals/cat")

Here's an example using a dataframe like this:

+--------------------------+
|        path              |
+==========================+
|"animals/cat/cat1.png"    |
+--------------------------+
|"animals/tiger/lion1.png" |
+--------------------------+
|"animals/tiger/tiger1.png"|
+--------------------------+
|"animals/dog/dog1.png"    |
+--------------------------+


.. highlight:: python
.. code-block:: python

    import pandas as pd

    animals_df = pd.DataFrame(
        [
            (f"animals/cat/cat1.jpg"),
            (f"animals/lion/lion1.jpg"),
            (f"animals/tiger/cat3.jpg"),
            (f"animals/dog/dog1.jpg")
        ],
        columns=["path", "label", "fold"],
    )

.. highlight:: python
.. code-block:: python

    vc.predict(animals_df)

Training a model and making predictions
---------------------------------------

You can also use a pre-trained model to adapt the model to your problem.
It is necessary to instantiate the VisionClassifier.
In this example only the last layer of the model will be trained for a multi-class
classification task.

.. highlight:: python
.. code-block:: python
    
    from aisee import VisionClassifier

    vc = VisionClassifier(
        model_name="vgg11_bn",
        num_classes=4,
        learning_method="freezed",
        task="single_label",
    )

The next step is to instantiate the Trainer with the attributes needed to perform the training.

To train an image classification model with our library, you will need to have a dataset.
The data could be in different formats: as a directory of folders or as a dataframe.
In this case the data will be found in a directory with the following structure:

.. code-block:: console

            ├── animals
              ├── cat
              │  ├── cat1.jpg
              |  ├── cat2.jpg
              │  └── ...
              ├── lion
              │  ├── lion1.jpg
              |  ├── lion2.jpg
              │  └── ...
              ├── tiger
              │  ├── tiger1.jpg
              |  ├── tiger2.jpg
              │  └── ...
              └── dog
                  ├── dog1.jpg
                  ├── dog2.jpg
                  └── ...

.. highlight:: python
.. code-block:: python

    from aisee import Trainer

    trainer = Trainer(
        base_model=vc,
        data=f"animals/",
        output_dir="test_trainer.pt",
        batch_size=8,
        checkpointing_metric="loss",
        shuffle=True,
    )

And finally call the train method of the Trainer class to train the model.

.. highlight:: python
.. code-block:: python

    trainer.train()


To make the predictions with our library, the data could be in different formats:
as a directory to a single image, as a directory of images or as a dataframe.

Here's an example using the directory to a single image:

.. highlight:: python
.. code-block:: python

    trainer.base_model.predict("animals/tiger/tiger1.jpg")

Here's an example using a directory of images:

.. highlight:: python
.. code-block:: python

    trainer.base_model.predict("animals/cat")

Here's an example using a dataframe like this:

+--------------------------+
|        path              |
+==========================+
|"animals/cat/cat1.png"    |
+--------------------------+
|"animals/tiger/lion1.png" |
+--------------------------+
|"animals/tiger/tiger1.png"|
+--------------------------+
|"animals/dog/dog1.png"    |
+--------------------------+


.. highlight:: python
.. code-block:: python

    import pandas as pd

    animals_df = pd.DataFrame(
        [
            (f"animals/cat/cat1.jpg"),
            (f"animals/lion/lion1.jpg"),
            (f"animals/tiger/cat3.jpg"),
            (f"animals/dog/dog1.jpg")
        ],
        columns=["path", "label", "fold"],
    )

.. highlight:: python
.. code-block:: python

    trainer.base_model.predict(animals_df)