# Gender Classification using Haarcascade and Gender Net.caffe Model

This project focuses on gender classification using Haarcascade for face detection and the Gender Net.caffe model for gender prediction. The implementation is done in Python with OpenCV and Caffe.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Gender classification is an essential task in many applications such as security systems, human-computer interaction, and demographic studies. This project utilizes Haarcascade for detecting faces in images and the Gender Net.caffe model to classify the detected faces as male or female.

## Features

- Face detection using Haarcascade.
- Gender classification using Gender Net.caffe model.
- Real-time gender classification using webcam feed.
- Easy-to-use command-line interface.

## Requirements

- Python 3.x
- OpenCV
- Caffe
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gender-classification.git
    ```
2. Navigate to the project directory:
    ```sh
    cd gender-classification
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Download the `deploy_gender.prototxt` and `gender_net.caffemodel` files and place them in the project directory.

## Usage

1. **Face Detection and Gender Classification on Images:**
    ```sh
    python classify_gender.py --image path_to_your_image.jpg
    ```
2. **Real-time Gender Classification using Webcam:**
    ```sh
    python classify_gender.py --webcam
    ```

### Command-Line Arguments

- `--image`: Path to the image file for gender classification.
- `--webcam`: Use webcam for real-time gender classification.

### Example

To classify gender from an image:
```sh
python classify_gender.py --image examples/sample.jpg
```
To use webcam for real-time gender classification:
```python classify_gender.py --webcam```

Results

Here are some sample results attached at my repo

