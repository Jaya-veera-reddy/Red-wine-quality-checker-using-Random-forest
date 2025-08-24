# Wine Quality Predictor 🍷

This project is a Streamlit dashboard for predicting the quality of red wine batches using a Random Forest Classifier trained on physicochemical properties.

## Features

- **Interactive Sliders:** Input wine properties using sidebar sliders.
- **Quality Prediction:** Predicts if a wine batch is "Good" or "Bad" and shows model confidence.
- **Feature Importances:** Visualizes which chemical properties most influence the prediction.
- **User Instructions:** Built-in guide for using the dashboard.

## Getting Started

### Prerequisites

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for required packages.

### Installation

1. Clone this repository.
2. Ensure `winequality-red.csv` is in the project folder.

### Install dependencies

```sh
pip install -r requirements.txt
```

### Run the app

```sh
streamlit run app.py
```

## Usage

1. Adjust the sliders in the sidebar to set wine properties.
2. Click **Predict Quality** to see the result and confidence.
3. View feature importances for model insights.

## Files

- [`app.py`](app.py): Main Streamlit dashboard.
- [`winequality-red.csv`](winequality-red.csv): Dataset.
- [`requirements.txt`](requirements.txt): Python dependencies.
