# Human Counter (ML Project WS 18/19)

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

## Goals

- Detect humans reliably
- Count humans

#### Bonus
- Use Camera Feed instead of static pictures

## Technologies

- [Filtered COCO Dataset](http://cocodataset.org/) (Common Objects in Context) with labled images of people in a context.
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [OpenCV with Tensorflow Models](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API) for reading and preprocessing images as well as visualizing the classification
- [Jupyter Notebook](https://jupyter.org/) for documentation

## Roadmap

- Install and setup Jupyter Notebook in the repository
  - document _here_ how to start
- Setup Python Project with OpenCV, tensorflow etc.
- Get the [example](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API) running (which uses the trained `ssd_mobilenet_v1_coco` model)
  - add comments, document the behaviour
- Analyze the recommended models
- Implement own model with the COCO dataset filtered by persons.
- Test own model with the COCO dataset filtered by persons.
- Add functionality to count the detected persons and visualize the counter
- Eventually add live classification with a webcam feed

