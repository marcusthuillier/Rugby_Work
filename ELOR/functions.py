import math
import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm
from typing import Tuple
import logging

# Set up logging configuration
logging.basicConfig(filename='elo_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to initialize ELO ratings and calculate the game margin
def set_ELO(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Convert scores to integer type
        df[["Home_Score", "Away_Score"]] = df[["Home_Score", "Away_Score"]].astype(int)
        
        # Calculate the margin of victory by subtracting Away_Score from Home_Score
        df["Margin"] = df.Home_Score - df.Away_Score
        
        # Remove rows with missing margin values
        df.dropna(subset=['Margin'], inplace=True)
        
        # Define the game outcome based on the margin: 1 for Home win, 0 for Away win, and 0.5 for draw
        df['Home_Away_Draw'] = np.where(df.Margin > 0, 1, np.where(df.Margin < 0, 0, 0.5))

        logging.info("ELO ratings initialized and game margin calculated.")
    except Exception as e:
        logging.error(f"Error in set_ELO: {e}")
    return df

# Function to calculate the probability of the home team winning
def Probability(teamhome: float, teamaway: float) -> float: 
    try:
        # Calculate win probability for the home team based on the rating difference
        return 1.0 / (1 + math.pow(10, (teamaway - teamhome) / 400)) 
    except Exception as e:
        logging.error(f"Error in Probability function: {e}")
        return 0.5  # Default probability in case of an error

# Function to update ELO ratings after a game
def EloRating(Rating_home: float, Rating_away: float, diff: int, K: int, d: float, home_advantage: float = 75) -> Tuple[float, float, float, float]: 
    try:
        # Adjust Home rating to simulate home advantage
        Rating_home += home_advantage
        
        # Calculate the win probabilities for Home and Away teams
        P_home = Probability(Rating_home, Rating_away) 
        P_away = Probability(Rating_away, Rating_home) 
        
        # Revert the temporary home adjustment for actual ELO calculations
        Rating_home -= home_advantage

        # Calculate margin of victory modifier
        mov = log(abs(diff) + 1)
        
        # Calculate correction factor for win probability difference
        cor = 2.2 / ((P_home - P_away) * 0.001 + 2.2)
        
        # Update ELO ratings based on the outcome (d) and computed factors
        Rating_home += cor * mov * K * (d - P_home) 
        Rating_away += cor * mov * K * ((1 - d) - P_away)

        # Log the updated ratings and probabilities
        logging.info(f"Updated Ratings - Home: {Rating_home}, Away: {Rating_away}, Prob Home: {P_home}, Prob Away: {P_away}")
        
        return (round(Rating_home, 6), round(Rating_away, 6), round(P_home, 3), round(P_away, 3))
    
    except Exception as e:
        logging.error(f"Error in EloRating function: {e}")
        return Rating_home, Rating_away, 0.5, 0.5  # Return defaults in case of an error

# Function to calculate win probability over a range of scores for visualizations or analysis
def WinPer() -> pd.DataFrame:
    tot = []
    # Loop over a range of rating scores in increments of 10 to calculate probabilities
    try:
        for x in tqdm(range(0, 2000, 10)):
            # Compute ELO update and probability for each score in the range
            _, _, homeprob, _ = EloRating(x, 1000, 10, 40, 1)
            
            # Store the rating difference (score) and win probability
            tot.append((x - 1000, homeprob))

        # Log completion of win probability calculation
        logging.info("Win probability range calculation completed.")
    except Exception as e:
        logging.error(f"Error in WinPer function: {e}")
        
    # Return results as a DataFrame with 'Score' and 'Prob' columns
    return pd.DataFrame(tot, columns=['Score', 'Prob'])

# Function to sequentially update ELO ratings across the dataset
def update_ELO(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize columns for win probabilities and updated ratings
    df["Prob_Home"] = pd.NA
    df["Prob_Away"] = pd.NA
    df["Home_Rating_Updated"] = pd.NA
    df["Away_Rating_Updated"] = pd.NA

    # Dictionary to store the most recent rating for each team
    team_ratings: dict[str, float] = {}

    # Iterate over each game in the dataset
    for x in tqdm(range(len(df))):
        try:
            # Retrieve or initialize the ratings for the home and away teams
            home_team = df.at[x, 'Home_Team']
            away_team = df.at[x, 'Away_Team']
            Ratinghome = team_ratings.get(home_team, 1500.0)
            Ratingaway = team_ratings.get(away_team, 1500.0)
            
            # Retrieve game margin and outcome type
            diff = int(df.at[x, 'Margin'])
            d = df.at[x, 'Home_Away_Draw']
            K = 40

            # Calculate updated ELO ratings and win probabilities for the current game
            Home_Rating_Updated, Away_Rating_Updated, Prob_Home, Prob_Away = EloRating(Ratinghome, Ratingaway, diff, K, d)

            # Update the DataFrame with calculated probabilities and updated ratings
            df.at[x, 'Prob_Home'] = Prob_Home
            df.at[x, 'Prob_Away'] = Prob_Away
            df.at[x, 'Home_Rating_Updated'] = Home_Rating_Updated
            df.at[x, 'Away_Rating_Updated'] = Away_Rating_Updated

            # Store updated ratings in dictionary for future games
            team_ratings[home_team] = Home_Rating_Updated
            team_ratings[away_team] = Away_Rating_Updated

            logging.info(f"Game {x}: Home Team: {home_team}, Away Team: {away_team}, Updated Ratings: {Home_Rating_Updated}, {Away_Rating_Updated}")

        except Exception as e:
            logging.error(f"Error in update_ELO function at game {x}: {e}")

    return df

def get_rank(df: pd.DataFrame) -> None:
    try:
        # Get a unique list of all team names from Home_Team and Away_Team columns
        names = set(df.Home_Team).union(df.Away_Team)
        ranking = []

        # Iterate over each team name to get the most recent ELO rating
        for name in names:
            try:
                # Get the last index where the team appears as Home or Away
                home_index = df[df.Home_Team == name].index[-1] if not df[df.Home_Team == name].empty else -1
                away_index = df[df.Away_Team == name].index[-1] if not df[df.Away_Team == name].empty else -1

                # Select the latest index and the corresponding ELO rating
                if home_index >= away_index:
                    elo_rating = round(float(df.at[home_index, 'Home_Rating_Updated']))
                else:
                    elo_rating = round(float(df.at[away_index, 'Away_Rating_Updated']))

                # Append the team name and ELO rating to the ranking list
                ranking.append((name, elo_rating))
                logging.info(f"Team: {name}, ELO: {elo_rating}")

            except Exception as e:
                logging.error(f"Error processing team '{name}': {e}")

        # Sort the ranking by ELO in descending order and save to CSV
        return pd.DataFrame(sorted(ranking, key=lambda x: x[1], reverse=True), columns=["Team", "ELO"])

    except Exception as e:
        logging.error(f"Error in get_rank function: {e}")

import logging
import pandas as pd

# Set up logging configuration
logging.basicConfig(filename='elo_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Get unique team names and years
        teams = set(df.Home_Team).union(df.Away_Team)
        results = []

        # Calculate expected vs. actual wins for each team and year
        for team in teams:
            df_team = df[(df.Home_Team == team) | (df.Away_Team == team)]
        
            expected, real = 0.0, 0.0

            # Calculate expected and actual wins for the current team and year
            for _, row in df_team.iterrows():
                try:
                    if row.Home_Team == team:
                        # Team is playing at home
                        expected += row.Prob_Home
                        real += row.Home_Away_Draw
                    else:
                        # Team is playing away
                        expected += row.Prob_Home
                        real += abs(row.Home_Away_Draw - 1)
                except Exception as e:
                    logging.error(f"Error processing game for team '{team}': {e}")

            # Append result for the team in the given year
            results.append((team, expected, real, real - expected))
            logging.info(f"Team: {team}, Expected Wins: {expected}, Actual Wins: {real}, Performance: {real - expected}")

        # Create and filter DataFrame for results
        df = pd.DataFrame(results, columns=["Team", "xWins", "Wins", "Performance"])
        df = df[df.xWins > 0]
        df.sort_values(by='Performance', ascending=False, inplace=True)
        logging.info("Team performance calculation completed and filtered.")

        return df

    except Exception as e:
        logging.error(f"Error in calculate_team_performance function: {e}")
        return pd.DataFrame(columns=["Year", "Team", "xWins", "Wins", "Performance"])  # Return empty DataFrame on error
    
# https://medium.com/towards-data-science/an-end-to-end-machine-learning-project-with-python-pandas-keras-flask-docker-and-heroku-c987018c42c7