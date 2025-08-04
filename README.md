# Sonimir
Welcome to Sonimir, an app that classifies audio (WAV) files. 

Sonimir is derived from _sonus_ (sound) + _mimir_ (Norse god of wisdom). It is a trained Convolutional Neural Network (CNN) based on ResNet deep learning architecture. This architecture was primarily designed to tackle the vanishing gradient problem, which causes gradients to become very small as they are backpropogated through many layers ofa  neural network. ResNet solves this by introducing skip connections (residual connections), allowing gradients to bypass certain layers in the network.

### Dataset

This CNN was trained on the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings consisting of 5-second .wav files organized into 50 semantical classes. Each class has about 40 labeled examples.

### Use

### Demo

### Tech Stack
Backend (Training)
  - Python 🐍
  - PyTorch 🔥
  - Pandas 🐼
  - Numpy ➕➗
  - TensorBoard 📊
  - Modal (cloud-based) ☁️🧪
    -  NVIDIA A10G w/ CUDA (training) ⚡🧠
    -  Serverless FastAPI endpoint (model inferencing) 🚀🧩
    
Frontend
  - React ⚛️
  - Express 🚂
  - Shadcn 🌿🧱 (UI)
  - TypeScript 🔷
  - AWS Amplify (hosting) ☁️📡

### Setup

Trained a Convolutional Neural Network (CNN) to classify .wav audio files in full-stack app (PyTorch, Next.js, React, Tailwind, Python)

- `pip install torch`
- `pip install modal`
- `modal setup`
- `pip install pandas`
