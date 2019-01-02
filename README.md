# Fruit Detection (ML Project WS 18/19)

## Report structure and proceeding

1. Introduction: Preprocess Dataset, see if Coco-Dataset would be a good choice too and compare accuracy between RGB and Grayscale → Jochen
2. Create a Multi Layer Perceptron (MLP) with Fruits-Dataset and document procedure → Tim
3. Visualize MLP with Tensorboard (train_acc, train_loss, val_acc, val_loss, graph) and try/tweak different parameters (change layers, neurons, overfitting, underfitting, ...) → Yannick
4. Create a Convolutional Neural Network (CNN) with Fruits-Dataset and optimize val_accuracy. Implement realtime detection with OpenCV → Passi
5. Improve realtime fruit detection and add model bounding boxes → Valentin


## Installation

### Anaconda virtual environment

All packages are managed through a [conda](https://conda.io/docs/) virtual environment named `human-counter`.
- Install it with `conda env create -f meta.yml`
- Activate it with `source activate human-counter` (`activate human-counter` on Windows)
- Deactivate it with `source deactivate` (`deactivate` on Windows)
- Install new packages with `conda install <NEW_PACKAGE> && conda env export > environment.yml`

### Jupyter Notebook

If you use ***Anaconda*** jupyter notebook is already installed. Else you can install it with pythons package manager ***pip***.

Therefore run following commands:

- As always check if the latest pip version is installed via `pip3 install --upgrade pip`
- Next install jupyter notebook via `pip3 install jupyter` for the latest version

You can now run jupyter notebook via terminal like this: `jupyter notebook` (don't forget to first change into the wanted directory). This will open your webbrowser with a overview of your directorys content. To create a new *.ipynb*-file click the *NEW*-button on the upper right corner and choose `Notebook:` → `Python 3`. This will open a new file where you can insert your code or markdown by choosing it in the upper mid dropdown-menu. To create a new cell click on the *plus*-button on the upper left. Don't forget so save your results all in a while!
