# 4990Regression

Instructions to run file:

1. `pipenv install`
2. `pipenv shell`
3. `pipenv run regression` or `python3 regression.py`

# 1. Problem Identification

Problem: Predicting attendance at NFL games <br>
Target Variable: Attendance <br>
Features / Independent Variables: Weather, Current Win Rate, Strength of Opponent <br>

# 2. Data Collection

Data sets taken from Kaggle: 
- [NFL Stadium Attendance](https://www.kaggle.com/datasets/sujaykapadnis/nfl-stadium-attendance-dataset) 
- [Weather data for NFL games](https://www.kaggle.com/datasets/tombliss/weather-data)

# 3. Model Development
Used a linear regression and polynomial regression model to predict attendance at NFL games.


# 4. Model Evaluation
<img width="558" alt="image" src="https://github.com/GuyWhoCode/4990Regression/assets/34526187/6d4c3cd4-277c-4b17-ad4f-e127761a5e73">

Based on the image shown above, the models show a low R^2 value. The R^2 value for the linear regression model is around `1.9 x 10^-5` while the R^2 value of the polynomial regression model is around `0.0028`. However, the linear regression model shows a significantly lower R^2 value than the polynomial regression model, meaning that it is a worse fit.

However, the R^2 values should be near one. This means that the regression models used might not be the most optimal models for predictions on this particular data.
