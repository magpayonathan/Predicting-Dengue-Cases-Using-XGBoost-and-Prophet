# Predicting-Dengue-Cases-Using-XGBoost-and-Prophet

# About
Dengue has been a recurring epidemic in the Philippines since 1954. It is transmitted by Aedes aegypti and Aedes albopictus mosquitoes, primarily during the rainy season from June to February. In 2025, the Philippines is experiencing a significant dengue outbreak, with cases rising sharply across the country, including in Quezon City. The local government has declared a dengue outbreak and implemented protective measures to curb the transmission rate. Previous efforts have attempted to forecast dengue outbreaks to support public health preparedness. This study aims to develop predictive models for weekly dengue cases, specifically XGBoost and Prophet, to forecast cases in Quezon City, Philippines, using parameters such as previous dengue cases, relative humidity, temperature (maximum and minimum), rainfall amount, wind speed, and wind direction. 

# Data Collection
Weekly data on dengue cases from 2015 to 2024 were obtained from the Quezon City Epidemiology and Surveillance Division (QCESD).   Climate variable data—including rainfall amount, maximum and minimum temperature, relative humidity, wind speed, and wind direction—were requested from the Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA) at their synoptic station located at the Science Garden, Quezon City. The dataset from PAGASA is in a daily format and covers the period from 2015 to 2023. 

# Model Setups
Three setups were created for both the XGBoost and Prophet models to evaluate their accuracy and performance. Setup 1 is used to predict the weekly dengue cases in 2024 without access to actual 2024 weather data. Setup 2 is used to predict weekly dengue cases in 2023 with access to actual 2023 weather data. Setup 3 is used to predict weekly dengue cases in 2023 without access to actual weather data in 2023.

| Setup 1 | Setup 2 | Setup 3 |
|----------|----------|----------|
| R1 C1    | R1 C2    | R1 C3    |
| R2 C1    | R2 C2    | R2 C3    |
| R3 C1    | R3 C2    | R3 C3    |
| R4 C1    | R4 C2    | R4 C3    |
| R5 C1    | R5 C2    | R5 C3    |
