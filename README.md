# Human Counter (ML Project WS 18/19)

## Installation

All packages are managed through a [conda](https://conda.io/docs/) virtual environment named `human-counter`.
- Install it with `conda env create -f meta.yml`
- Activate it with `source activate human-counter` (`activate human-counter` on Windows)
- Deactivate it with `source deactivate` (`deactivate` on Windows)

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

