'''
IMPORTANT NOTE: Model Evaluation information can be found by clicking on the README.md file
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Select specific columns from the csv files based on how relevant they are to training the model
GAME_WEATHER = pd.read_csv(
    'kaggleData/games_weather.csv', header=0,
    usecols=["game_id", "Temperature", "EstimatedCondition"])

GAMES_DATA = pd.read_csv(
    'kaggleData/games.csv', header=0,
    usecols=["game_id", "Season", "StadiumName"])

ATTENDANCE = pd.read_csv(
    'kaggleData/attendance.csv',
    header=0, usecols=["team", "year", "weekly_attendance"])

STANDINGS = pd.read_csv(
    'kaggleData/standings.csv', header=0,
    usecols=["team", "year", "simple_rating"])

# Merge game weather and data together based on game_id
COMBINED_GAME_WEATHER_DATA = pd.merge(GAME_WEATHER, GAMES_DATA, on='game_id')

# Delete the 'game_id' column
COMBINED_GAME_WEATHER_DATA = COMBINED_GAME_WEATHER_DATA.drop('game_id', axis=1)

# Merge attendance and standings data together based on team and year
COMBINED_ATTENDANCE_DATA = pd.merge(ATTENDANCE, STANDINGS, on=['team', 'year'])

'''
Stadium.json data provided through https://gist.github.com/brianhatchl/59d99872a9cfc0e126211192673991b8?short_path=e22db83

Note: had to be modified to have the stadium name mapped to the team city
'''
with open('stadium.json') as f:
    # Normalizes the data by replacing the stadium names with the team names using a dictionary
    stadium_data = json.load(f)
    COMBINED_GAME_WEATHER_DATA['StadiumName'] = COMBINED_GAME_WEATHER_DATA['StadiumName'].replace(
        stadium_data)
    COMBINED_GAME_WEATHER_DATA.rename(
        columns={'StadiumName': 'team'}, inplace=True)

    # Further normalize data by replacing the 'Season' column with 'year'
    COMBINED_GAME_WEATHER_DATA.rename(columns={'Season': 'year'}, inplace=True)

    # Create the actual training data used to train the model
    TRAINING_DATA = pd.merge(COMBINED_GAME_WEATHER_DATA,
                             COMBINED_ATTENDANCE_DATA, on=['team', 'year'])
    TRAINING_DATA.fillna(0, inplace=True)

    print(TRAINING_DATA.head())

    # Train linear regression model to predict weekly attendance based on temperature and simple rating
    X = TRAINING_DATA[['Temperature', 'simple_rating']]
    Y = TRAINING_DATA['weekly_attendance']
    linear_regression_model = LinearRegression().fit(X, Y)

    # Print the model's coefficients
    print(f"Linear Model Coefficients: {linear_regression_model.coef_}")
    print(f"Linear Model Intercept: {linear_regression_model.intercept_}")

    # Sample prediction
    test1 = pd.DataFrame([[70, 0.5]], columns=['Temperature', 'simple_rating'])
    test2 = pd.DataFrame(
        [[90, 5]], columns=['Temperature', 'simple_rating'])
    test3 = pd.DataFrame([[50, 10]], columns=['Temperature', 'simple_rating'])
    test4 = pd.DataFrame([[80, 10]], columns=['Temperature', 'simple_rating'])

    print("Linear Regression Model Predictions:")
    print(
        f"Temperature of 70 degrees and a team rating of 0.5 yields {int(linear_regression_model.predict(test1)[0])} fans")
    print(
        f"Temperature of 90 degrees and a team rating of 5 yields {int(linear_regression_model.predict(test2)[0])} fans")
    print(
        f"Temperature of 50 degrees and a team rating of 10 yields {int(linear_regression_model.predict(test3)[0])} fans")
    print(
        f"Temperature of 80 degrees and a team rating of 10 yields {int(linear_regression_model.predict(test4)[0])} fans")

    print("\n\n\n\n")

    # Create a polynomial regression model
    DEGREE = 4
    poly_model = make_pipeline(
        PolynomialFeatures(degree=4), LinearRegression())

    poly_model.fit(X, Y)
    print("Polynomial Regression Model Predictions:")
    print(
        f"Temperature of 70 degrees and a team rating of 0.5 yields {int(poly_model.predict(test1)[0])} fans")
    print(
        f"Temperature of 90 degrees and a team rating of 5 yields {int(poly_model.predict(test2)[0])} fans")
    print(
        f"Temperature of 50 degrees and a team rating of 10 yields {int(poly_model.predict(test3)[0])} fans")
    print(
        f"Temperature of 80 degrees and a team rating of 10 yields {int(poly_model.predict(test4)[0])} fans")

    from sklearn.metrics import r2_score

    # Assume you have some test data and predictions from your models
    y_test = ...
    linear_predictions = ...
    poly_predictions = ...

    # Calculate R-squared for the linear regression model
    linear_r2 = r2_score(y_test, linear_predictions)

    # Calculate R-squared for the polynomial regression model
    poly_r2 = r2_score(y_test, poly_predictions)

    print(f"Linear Regression R^2: {linear_r2}")
    print(f"Polynomial Regression R^2: {poly_r2}")
