AI Image Captioning – CNN + LSTM

This project is an end-to-end deep learning solution for automatically generating captions for images. It combines Computer Vision and Natural Language Processing by using a Convolutional Neural Network (CNN) for feature extraction and an LSTM (Long Short-Term Memory) network for sequence generation.

#What it Does

Feature Extraction: Uses a pre-trained InceptionV3 model to extract high-level visual features from images.

Caption Generation: An LSTM network generates natural-language descriptions based on the extracted features.

Interactive Demo: A Streamlit web app lets users upload an image and get an AI-generated caption.


## Installation
```bash
pip install -r requirements.txt
```

## Training
Requires preprocessing of Flickr8k or MS COCO dataset before training.

```bash
cd src
python train.py
```

## Run the Web App
```bash
streamlit run app.py
```

