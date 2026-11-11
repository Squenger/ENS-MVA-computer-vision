## Object recognition and computer vision 2026/2027
### Assignment 3: Synthetic CIFAR-10 image classification

#### Requirements

1. Install PyTorch from [pytorch.org](https://pytorch.org/).
2. Run the following command to install the additional dependencies:

   ```bash
   pip install -r requirements.txt
   ```

#### Dataset

We will be using the ten-class CIFAR-10 dataset. The training and validation images come from standard CIFAR-10, while the test images come from a synthetic, distribution-shifted target domain. Download `train.npz`, `validation.npz`, `test.npz`, and `sample_submission.csv` from the private Kaggle competition. The test image labels are not provided.

#### Training and validating your model

You can use either the notebook or the command-line script.

To use the notebook, open `main_notebook.ipynb`. On Kaggle, attach the competition data and update `INPUT_DIR` if necessary. For local use, set:

```bash
export RECVIS_INPUT_DIR=/path/to/competition/data
export RECVIS_OUTPUT_DIR=/path/to/output
jupyter lab main_notebook.ipynb
```

To train from the command line, run:

```bash
python main.py --data /path/to/competition/data --output runs/resnet18 --model resnet18 --epochs 10
```

For an ImageNet-pretrained model, add `--pretrained`. Run `python main.py --help` to see all options.

Modify `main_notebook.ipynb`, `main.py`, `model.py`, and `data.py` for your assignment, with the aim of improving the validation and Kaggle scores. Both entry points provide a ResNet-18 baseline and support pretrained ResNets, ViT, and linear probing. The notebook also describes an optional OpenCLIP baseline.

#### Evaluating your model on the test set

Run the submission cell at the end of the notebook, or let `main.py` finish training. Both generate `submission.csv`, which you can upload to the private Kaggle competition website. The file must contain one integer label from 0 to 9 for every test ID.

#### Rules

Open-source models, pretrained checkpoints, and external data are allowed if they are fully documented in the report. Proprietary or closed models and API-only services are prohibited. Do not obtain, reconstruct, manually annotate, or otherwise access the hidden test labels.

#### Acknowledgments

The source-domain data is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

This starter is adapted from the [2024 RecVis Assignment 3 repository](https://github.com/willowsierra/recvis24_a3).
