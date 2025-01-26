# Atmosphere ML 🧠

The machine learning backbone of [Atmosphere](https://github.com/maxrross/atmosphere), a comprehensive environmental health dashboard. This repository contains our custom-built spatiotemporal ML model for air quality predictions and the data preprocessing pipeline.

## Overview

We built a deep neural network that predicts air quality metrics across the United States. The model takes in location data and historical pollution measurements to forecast:

- Pollutant concentrations (O₃, CO, SO₂, NO₂)
- AQI values for each pollutant
- Peak pollution hours
- Overall air quality risk levels

## Data Pipeline

Our preprocessing pipeline handles:

1. Cleaning and normalizing EPA pollution data (2000-2023)
2. Geocoding locations using the Census API
3. Feature engineering for temporal patterns
4. Data standardization for model input

## Model Architecture

We use a deep neural network with:

- 7 input features (location, time, historical measurements)
- Multiple dense layers with ReLU activation
- Dropout for regularization
- Custom loss function optimized for AQI prediction

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/blakerand/atmosphere-ml.git
cd atmosphere-ml
```

2. Install dependencies:

```bash
cd pollution_prediction
npm install
```

3. Run predictions:

```bash
node predict.js
```

## Dataset

We trained on the [EPA Air Quality Dataset](https://www.kaggle.com/datasets/guslovesmath/us-pollution-data-200-to-2022/data), which includes:

- 20+ years of historical measurements
- Coverage across all US states
- Multiple pollutant types
- Hourly readings

## Built With

- TensorFlow.js - Model training and deployment
- Node.js - Data preprocessing
- Census Geocoding API - Location data processing

## Hackathon Project

This model was developed as part of Atmosphere for Swamphacks. Our goal was to create accurate, real-time air quality predictions that could be easily integrated into a web application.
