# Description 📊

This project analyzes Belgian real estate data sourced from **[ImmoVlan](https://immovlan.be/en)**.

The goal is to clean, structure, and explore the dataset to uncover market trends, identify price drivers, and prepare the data for future predictive modeling.

# Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Cleaning Pipeline](#cleaning-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

# Project Overview

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

# Cleaning Pipeline

### Key steps:

1. Remove duplicates
- Based on unique property IDs.

2. Trim whitespace
- Fixes formatting (e.g., " Brussels " → "Brussels").

3. Convert data types
- price, living_area, facades, and number_rooms → numeric
- postal_code → string
- province → categorical

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
│   └── Hamideh
│             └──analysis.ipynb
├── output/
│   ├── images.png
│   ├── overview.png
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

Include the following libraries :

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

# The team
This project is part of AI & Data Science Bootcamp training at **`</becode>`** and it
  by :

- Sandrine Herbelet  [LinkedIn](https://www.linkedin.com/in/) | [Github](https://github.com/Sandrine111222)
- Hamideh Baggali [LinkedIn](https://www.linkedin.com/in/hamideh-be/) |[Github](https://github.com/Hamideh-B-H)
- Intan K.Wardhani [Github](https://github.com/intanwardhani)
- Welederufeal Tadege [LinkedIn](https://www.linkedin.com/in/) | [Github](https://github.com/welde2001-bot)


It was  done under the supervision of our coach ***Vanessa Rivera Quinones***
