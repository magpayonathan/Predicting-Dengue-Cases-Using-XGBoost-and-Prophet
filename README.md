# Predicting-Dengue-Cases-Using-XGBoost-and-Prophet

# About
Dengue has been a recurring epidemic in the Philippines since 1954. It is transmitted by Aedes aegypti and Aedes albopictus mosquitoes, primarily during the rainy season from June to February. In 2025, the Philippines is experiencing a significant dengue outbreak, with cases rising sharply across the country, including in Quezon City. The local government has declared a dengue outbreak and implemented protective measures to curb the transmission rate. Previous efforts have attempted to forecast dengue outbreaks to support public health preparedness. This study aims to develop predictive models for weekly dengue cases, specifically XGBoost and Prophet, to forecast cases in Quezon City, Philippines, using parameters such as previous dengue cases, relative humidity, temperature (maximum and minimum), rainfall amount, wind speed, and wind direction. 

# Data Collection
Weekly data on dengue cases from 2015 to 2024 were obtained from the Quezon City Epidemiology and Surveillance Division (QCESD).   Climate variable data—including rainfall amount, maximum and minimum temperature, relative humidity, wind speed, and wind direction—were requested from the Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA) at their synoptic station located at the Science Garden, Quezon City. The dataset from PAGASA is in a daily format and covers the period from 2015 to 2023. 

# Model Setups
Three setups were created for both the XGBoost and Prophet models to evaluate their accuracy and performance. The details of the three setups are shown below.
| Setup 1 | Setup 2 | Setup 3 |
|----------|----------|----------|
| Predicting 2024 Weekly Cases    | Predicting 2023 Weekly Cases    | Predicting 2023 Weekly Cases    |
| Trsining set: 2015-2023    | Training set: 2015-2022    | Training set: 2015-2022    |
| No access to 2024 weather data    | With access to actual 2023 weather data    | No access to 2023 weather data    |
| Evaluated using actual 2024 cases    | Evaluated using actual 2023 cases    | Evaluated using actual 2023 cases    |

# Instructions
Hey
