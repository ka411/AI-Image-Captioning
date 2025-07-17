AI Image Captioning – CNN + LSTM

This project is an end-to-end deep learning solution for automatically generating captions for images. It combines Computer Vision and Natural Language Processing by using a Convolutional Neural Network (CNN) for feature extraction and an LSTM (Long Short-Term Memory) network for sequence generation.


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

