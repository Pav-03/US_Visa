# 🇺🇸 US Visa Prediction MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![GithHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

<br/>
<p align="center">
  <a href="#about-the-project">About The Project</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#built-with">Built With</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#deployment">Deployment</a>
</p>
</div>

---

## 📖 About The Project

The **US Visa Prediction MLOps Pipeline** is an end-to-end Machine Learning project designed to predict the status of US Visa applications ("Certified" or "Denied"). 

This project goes beyond simple modeling by implementing a robust, production-ready **MLOps architecture**. It automates the entire lifecycle from data ingestion to model deployment, ensuring scalability, reproducibility, and continuous delivery.

### ✨ Key Features
*   **Full MLOps Pipeline**: Data Ingestion, Validation, Transformation, Model Training, Evaluation, and Pushing.
*   **Automated CI/CD**: Seamless deployment to **Google Cloud Run** using GitHub Actions.
*   **Premium Web Interface**: A modern, responsive web application built with **FastAPI** and custom CSS for real-time predictions.
*   **Containerized**: Fully Dockerized for consistent environments across development and production.
*   **Experiment Tracking**: Modular component design allows for easy integration with tools like MLflow or DVC.

## 🏗 Architecture

The project follows a component-based modular architecture:

1.  **Data Ingestion**: Fetches data from MongoDB.
2.  **Data Validation**: Validates schema and checks for drift using Evidently.
3.  **Data Transformation**: Handles cleaning, feature engineering (`company_age`), and preprocessing.
4.  **Model Trainer**: Trains multiple models (KNeighbors, XGBoost, etc.) and selects the best one.
5.  **Model Evaluation**: Compares the trained model with the current production model.
6.  **Model Pusher**: Deploys the best model to the serving registry (S3/GCS).

## 🛠 Built With

*   **Language**: Python 3.8+
*   **Web Framework**: FastAPI, Jinja2
*   **ML Libraries**: Scikit-learn, Pandas, NumPy, Imbalanced-learn
*   **Data Validation**: Evidently AI
*   **Infrastructure**: Docker, Google Cloud Platform (Cloud Run, Artifact Registry)
*   **CI/CD**: GitHub Actions

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   MongoDB Atlas Account
*   Google Cloud Account (for deployment)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Pav-03/US_Visa.git
    cd US_Visa
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables**
    Create a `.env` file or export variables:
    ```bash
    export MongoDB_url="your_mongodb_connection_string"
    export AWS_ACCESS_KEY_ID="your_aws_access_key"
    export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
    ```

4.  **Run Locally**
    ```bash
    python app.py
    ```
    Access the app at `http://localhost:8080`

## ☁️ Deployment

The project is configured for **Continuous Deployment** to Google Cloud Run.

1.  **Setup GCP**: Create a Project and enable Cloud Run & Artifact Registry APIs.
2.  **Configure Secrets**: Add `GCP_PROJECT_ID` and `GCP_CREDENTIALS` (JSON Service Account Key) to your GitHub Repository Secrets.
3.  **Push to Main**: The GitHub Actions workflow will automatically build the Docker image and deploy it to Cloud Run.

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---
<div align="center">
    <b>Star this repo if you find it useful! ⭐</b>
</div>