# 🌍 Satellite Surface Typography Classifier

An end-to-end deep learning web application that classifies satellite imagery into 10 distinct terrain categories in real-time. 

## 📊 Business Value & Use Case
This application allows geospatial analysts, operations managers, and environmental planners to instantly classify land use (e.g., urban development, agricultural zoning, or water resources) without needing to write code or manually process raw satellite data. By decoupling the heavy model training from a lightweight web interface, it provides a scalable, user-friendly tool for rapid spatial decision-making.

## 🛠️ Technical Architecture

* **Frontend Dashboard (Python Shiny):** An interactive, reactive web interface built with Shiny for Python. It handles user uploads and renders real-time probability distributions using Pandas.
* **Backend Inference (PyTorch):** A custom 10-class Convolutional Neural Network (CNN) engineered with Residual Blocks. The model weights are loaded directly onto the CPU for lightning-fast, cost-effective web inference.
* **Offline Training Pipeline:** The model was trained using rigorous data augmentation (color jittering, Gaussian blur, random rotations), Batch Normalization, and a Cosine Annealing Warm Restarts learning rate scheduler to prevent overfitting and ensure high generalization accuracy.

## 📂 Repository Structure
* `/app` - Contains the `app.py` script for the Shiny web dashboard and inference logic.
* `/model` - Contains the decoupled PyTorch CNN architecture and the training pipeline scripts.
* `/assets/plots` - Exploratory Data Analysis (EDA) charts showcasing average pixel and brightness distributions across the dataset.

## 🚀 How to Run Locally

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

   <img width="1920" height="1039" alt="image" src="https://github.com/user-attachments/assets/b6a38581-f4ff-4316-a916-fda6bdb8279c" />
