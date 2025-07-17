# AI Image Captioning

This project uses a CNN (InceptionV3) for feature extraction and an LSTM network for generating captions for images.

## Features
- Feature extraction using pre-trained InceptionV3
- LSTM-based text generation
- Streamlit app to upload images and get captions

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

