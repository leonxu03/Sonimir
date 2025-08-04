# Sonimir
Welcome to Sonimir, an app that classifies audio (WAV) files. 

Sonimir is derived from _sonus_ (sound) + _mimir_ (Norse god of wisdom). It is a trained Convolutional Neural Network (CNN) based on ResNet deep learning architecture. This architecture was primarily designed to tackle the vanishing gradient problem, which causes gradients to become very small as they are backpropogated through many layers ofa  neural network. ResNet solves this by introducing skip connections (residual connections), allowing gradients to bypass certain layers in the network.

## Dataset

This CNN was trained on the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings consisting of 5-second .wav files organized into 50 semantical classes. Each class has about 40 labeled examples.

## Use

## Demo

## 💻 Tech Stack

### 🧠 Backend (Training)
- **Python** 🐍  
- **PyTorch** 🔥  
- **Pandas** 🐼  
- **NumPy** ➕➗  
- **TensorBoard** 📊  
- **Modal (Cloud-Based)** ☁️🧪  
  - NVIDIA A10G w/ CUDA (Training) ⚡🧠  
  - Serverless FastAPI Endpoint (Model Inferencing) 🚀🧩  

### 🎨 Frontend
- **React** ⚛️
- **Tailwind CSS** 🎨
- **Express** 🚂  
- **Shadcn (UI)** 🌿🧱  
- **TypeScript** 🔷  
- **AWS Amplify (Hosting)** ☁️📡  

## Setup

### 🧠 Backend
Install dependencies:
```
pip install -r requirements.txt
```

Modal setup:
```
modal setup
```

Hit modal endpoint from local:
```
modal run main.py
```

Deploy backend:
```
modal deploy main.py
```

### 🏋️ Training
Deploy training script:
```
modal deploy train.py
```

Execute training script on cloud:
```
modal run train.py
```

### 🎨 Frontend
Install dependencies:
```
cd sonimir && npm install
```

Run locally:
```
npm run dev
```

Build:
```
npm run build
```
