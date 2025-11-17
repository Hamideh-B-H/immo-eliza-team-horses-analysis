#  🔔 Immo Eliza – Real Estate Data Cleaning & Analysis

<img src="images/banner.png" width="100%" />

This project analyzes Belgian real estate data scraped from the Immo Eliza platform.

The goal is to clean, structure, and explore the dataset to uncover market trends, identify price drivers, and prepare the data for future predictive modeling.

# Project Overview

<img src="images/overview.png" width="70%" />

The project covers:

- Cleaning and standardizing real estate data

- Removing outliers and invalid entries

- Performing descriptive and exploratory analysis

- Visualizing regional, structural, and price-based trends

- Identifying the major factors influencing property prices

# Features

- Comprehensive data cleaning pipeline

- Automatic handling of:

    - Missing values

    - Wrong data types

    - Bad formatting

    - Duplicates

- Computation of new variables (e.g., price per m², size bins)

- Regional and municipal analysis

- Price distribution and correlation studies

- High-quality Matplotlib/Seaborn visualizations

# Dataset

The cleaned dataset contains:

- 15,000+ property listings

- 17 features, including:

    - Price

    - Living area

    - Build year

    - Number of rooms

    - Number of facades

    - Property type

    - Province & region

    - Price per square meter

Cleaning Pipeline
<img src="images/cleaning.png" width="70%" />

# Key steps:

1. Remove duplicates

Based on unique property IDs.

2. Trim whitespace

Fixes formatting (e.g., " Brussels " → "Brussels").

3. Convert data types

price, living_area, facades, and number_rooms → numeric

postal_code → string

province → categorical

4. Handle missing and invalid values

- Replace malformed text values with NaN

- Remove rows with missing province

- Drop entries with impossible values

5. Generate additional variables

- price_per_m2 = price / living_area

- Binning (living area, number of rooms, price categories)

6. Remove outliers (IQR method)

Applied to:

- price

- living_area

# Exploratory Data Analysis

<img src="images/eda.png" width="70%" />

Full EDA includes:

- Distribution of prices & living areas

- Correlation matrix (heatmap)

- Price vs living area

- Price vs number of rooms

- Facades vs living area

- Regional comparison of price per m²

- Top 10 most/least expensive municipalities

- Property counts by price segments

# Key Insights

- Brussels is the most expensive region; Wallonia is the most affordable.

- Living area is the strongest predictor of price.

- Detached homes (4 facades) are larger and significantly more expensive.

- Price per m² varies heavily across municipalities.

- Several extreme outliers exist and strongly influence mean values.

 # 🧱 Project Structure

``` bash
IMMO_EELIZA_TEAM_HORSES_ANALYSIS/
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── analysis/
│   └── analysis.ipynb
│
├── images/
│   ├── banner.png
│   ├── cleaning.png
│   ├── overview.png
│   └── eda.png
│
├── src/
│   ├── cleaning.py
│   ├── analysis.py
│   └── visualization.py
│
├── README.md
└── requirements.txt
```

# Requirements

Include the following in requirements.txt:

```
pandas
numpy
matplotlib
seaborn

```
# How to Run

## ⚙️ 1. Clone the repository



git clone https://github.com/Hamideh-B-H/immo-eliza-team-horses-analysis

## 🧩 2. Install dependencies

```  

pip install -r requirements.txt

```
## 3. Open the analysis notebook

- jupyter notebook notebooks/analysis.ipynb

## 🕹️ 4. Run the cleaning script (optional)



python src/cleaning.py

# Contributors

We are Sandrine Herbelet, Hamideh Baggali, 

Welederufeal Tadege and Intan K.Wardhani

 and are AI & Data Science trainees 

at Becode.
