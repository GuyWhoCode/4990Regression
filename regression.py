import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# Load the csv data using only the game_id, Temperature, and EstimatedCondition columns
GAME_WEATHER = pd.read_csv(
    'kaggleData/games_weather.csv', header=0,
    usecols=["game_id", "Temperature", "EstimatedCondition"])

# CSV data with only the game_id, Season, and StadiumName columns
GAMES_DATA = pd.read_csv(
    'kaggleData/games.csv', header=0,
    usecols=["game_id", "Season", "StadiumName"])

# CSV data with only the team, year, week, and weekly_attendance columns
ATTENDANCE = pd.read_csv(
    'kaggleData/attendance.csv',
    header=0, usecols=["team", "year", "week", "weekly_attendance"])

# CSV data with only the team, year, and simple_rating columns
STANDINGS = pd.read_csv(
    'kaggleData/standings.csv', header=0,
    usecols=["team", "year", "simple_rating"])

# Merge game weather and data together based on game_id
COMBINED_GAME_WEATHER_DATA = pd.merge(GAME_WEATHER, GAMES_DATA, on='game_id')

# Delete the 'game_id' column
COMBINED_GAME_WEATHER_DATA = COMBINED_GAME_WEATHER_DATA.drop('game_id', axis=1)

# Merge attendance and standings data together based on team and year
COMBINED_ATTENDANCE_DATA = pd.merge(ATTENDANCE, STANDINGS, on=['team', 'year'])

with open('stadium.json') as f:
    stadium_data = json.load(f)
    COMBINED_GAME_WEATHER_DATA['StadiumName'] = COMBINED_GAME_WEATHER_DATA['StadiumName'].replace(
        stadium_data)
    COMBINED_GAME_WEATHER_DATA.rename(
        columns={'StadiumName': 'team'}, inplace=True)
    # Normalizes the data by replacing the stadium names with the team names using a dictionary

    COMBINED_GAME_WEATHER_DATA.rename(columns={'Season': 'year'}, inplace=True)
    # Further normalize data by replacing the 'Season' column with 'year'

    TRAINING_DATA = pd.merge(COMBINED_GAME_WEATHER_DATA,
                             COMBINED_ATTENDANCE_DATA, on=['team', 'year'])
    print(TRAINING_DATA.head())
