# Project Description: Creating a regression model using Sci-Kit Learn's Random Forest Regressor machine learning model.
# Predicting an NBA statistic for a college player based on their college statistics.
# The college player and nba player data is extensively cleaned to ensure accurate and up-to-date information
# is passed through the model.
# The model trains the data and provides a R^2 score and MSE value.
# The model then finds the 10 most accurate predictors and allows the user to input the corresponding value
# per predictor.
# The model then does the calculations and returns the final prediction for the predicted NBA statistic.
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import operator

data_of_nba_players = 'players.csv'
list_of_active_NBA_players = []
names_of_active_NBA_players = []
# Opens the csv that contains the stats for all the active nba players and adds each player to a list
# called list_of_active_NBA_players. This list is then iterated through and the first value at the first index
# of each item, the name, is added to the list names_of_active_NBA_players.
with open(data_of_nba_players, 'r', newline='') as infile:
    reader = csv.reader(infile)

    for row in reader:
        list_of_active_NBA_players.append(row)
    for cleaned_row1 in list_of_active_NBA_players:
        names_of_active_NBA_players.append(cleaned_row1[0])


all_college_players = 'CollegeBasketballPlayers2009-2021.csv'
cleaned_college_players = 'cleaned_college_players.csv'
college_players = []
acceptable_conferences = ['ACC', 'B12', 'B10', 'SEC', 'BE', 'P12']
# Outputs a csv file that omits the college players who played before the year of 2015, are not in a top 6 college
# conference, and whose name isn't in the list of active nba players. This results in a file that contains simply the
# players who fit the desired qualifications.
with open(all_college_players, 'r', newline='') as infile, open(cleaned_college_players, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    first_row = next(reader)
    column_labels = first_row

    for row in reader:
        college_players.append(row)
    for cleaned_row in college_players:
        for conference in acceptable_conferences:
            if cleaned_row[2] == conference and int(cleaned_row[31]) >= 2015 and cleaned_row[0] in names_of_active_NBA_players:
                writer.writerow(cleaned_row)


NBA_player_stats = "2022-2023 NBA Player Stats - Regular.csv"
encoding = "utf-16"
NBA_player_stats_list = []
NBA_active_names2 = []
# Creates another list of active NBA Player names coming from the list in which the data from the response variable
# will be coming from to ensure the data sets contain the correct information. This input csv is encoded with utf-16
# so that must be included in the open statement. Rather than a comma separating the data in the file it's a semicolon,
# so the delimiter is assigned to be a semicolon.
with open(NBA_player_stats, 'r', newline='', encoding=encoding) as infile:
    reader = csv.reader(infile, delimiter=';')

    for row in reader:
        NBA_player_stats_list.append(row)
    for cleaned_row1 in NBA_player_stats_list[1:]:
        NBA_active_names2.append(cleaned_row1[1])


Clean_college_data = []
Cleaned_college_players_in_NBA = "CollegePlayersActiveInNBA.csv"
data_college_in_NBA = []
# Creates the final output csv called CollegePlayersActiveInNBA that contains the statistics of
# all active NBA players that were in the cleaned_college_players list. The first index of each row in
# Clean_college_data is checked to see if it is in the list of active nba names and the row is added
# to the output csv file if this boolean evaluates to true.
with open(cleaned_college_players, 'r', newline='') as infile, open(Cleaned_college_players_in_NBA, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        Clean_college_data.append(row)
    for cleaned_row in Clean_college_data:
        if cleaned_row[0] in NBA_active_names2:
            data_college_in_NBA.append(cleaned_row)
            writer.writerow(cleaned_row)


Possible_response_variables = []
# Iterates through the first row in the NBA_player_stats file to create a list of possible response variables that
# the user can choose from for regression analysis.
with open(NBA_player_stats, 'r', newline='', encoding=encoding) as infile:
    reader = csv.reader(infile, delimiter=';')

    first_row2 = next(reader)
    for label in first_row2:
        Possible_response_variables.append(label)
# Print out these possible response variables so the user knows the options to choose from.
print(Possible_response_variables)
# Prompts the user to input a response variable from the list displayed.
Response_variable = input("Choose a response variable from this list: ")
# For loop that creates a variable called response index so the correct response variable can be added to the
# list of players later on.
k = 0
Response_index = 0
for label in Possible_response_variables:
    if Response_variable == label:
        Response_index = k
    k += 1

# Adds the desired stat for the response variable to each player. Ensures the correct player receives the
# correct statistic by checking that the player name in data_college_in_NBA is equivalent to the player name
# in the respective row in the NBA_player_stats input file.
with open(NBA_player_stats, 'r', newline='', encoding=encoding) as infile:
    reader = csv.reader(infile, delimiter=';')

    for row in reader:
        for player in data_college_in_NBA:
            if player[0] == row[1]:
                # This call to append is where the response variable is added, and it's added to the end of each row
                # meaning it is the last index for each player row.
                player.append(row[Response_index])

cleaned_data_college_in_NBA = []
# Cleans the data so that there is no more null elements and no more elements that contain letters
for statistics in data_college_in_NBA:

    if not any(stat == "" for stat in statistics):
        cleaned_data_college_in_NBA.append(statistics)

# Ensures that all the columns that contain qualitative variables are removed from the data.
cleaned_data_college_in_NBA2 = [[element for element in row if not any(c.isalpha() for c in str(element))] for row in cleaned_data_college_in_NBA]

# Making sure every element is a float for the regression model.
cleaned_data_as_float = [[float(element) for element in statistics] for statistics in cleaned_data_college_in_NBA2]
fully_cleaned_data_as_float = []
# Ensures that there is no variation in length by row because it would create errors in the regression model.
# Does so by chopping off the data that the two different length rows do not have in common.
for statistics in cleaned_data_as_float:
    if len(statistics) == 60:
        fully_cleaned_data_as_float.append(statistics[:58] + [statistics[-1]])
    if len(statistics) == 62:
        fully_cleaned_data_as_float.append(statistics[:58] + [statistics[-1]])


clean_column_labels = []
# Ensures that the column labels list only contains the labels for the columns present in the data set.
for element in column_labels:
    if element != "player_name" and element != "team" and element != "conf" and element != "yr" and element != "ht" and element != "type" and element != "" and element != "num" and element != "pfr" and element != "year" and element != "pid" and element != "pick":
        clean_column_labels.append(element)

# Further cleaning of the data was required and handled by the following process.
# Indices of the labels that are kept in the model.
columns_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

# Created a new list with only the labels listed in the previous assignment.
cleaned_data_as_float_filtered = [[row[i] for i in columns_to_keep] for row in fully_cleaned_data_as_float]

# Success in NBA Context is displayed by the chosen response variable
# Now training a model that uses the college statistics to predict minutes per game
# First using Random Forest to determine the most accurate predictors for predicting the desired response variable
# Creating a dictionary to store MSE values for each variable
mse_results = {}

# Iterating through the explanatory variables in the data.
for col_index in range(len(fully_cleaned_data_as_float[0])-1):
    # Extract the current column as a single-feature dataset
    X_single = [row[col_index] for row in fully_cleaned_data_as_float]
    X_single = np.array(X_single).reshape(-1, 1)
    y = [row[-1] for row in fully_cleaned_data_as_float]

    # Splitting the data into training sets for X and Y for each column.
    X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

    # Creating a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=300, random_state=42)

    # Fitting the model for this column
    model.fit(X_train, y_train)

    # Predicting using the trained model
    y_prediction = model.predict(X_test)

    # Calculating MSE for this column
    mse = mean_squared_error(y_test, y_prediction)

    # Storing MSE value in the dictionary
    mse_results[col_index] = mse

# Sorting the MSE values from lowest to highest.
sorted_mse = sorted(mse_results.items(), key=operator.itemgetter(1))

# Obtaining the 10 lowest MSE values because these will be the 10 most accurate at predicting
# the response variable.
top_10_lowest_mse = sorted_mse[:10]

# Creating a list of column indexes that correspond to the 10 chosen MSE values that are required for analysis.
col_index_identifier = []
for col_index, mse in top_10_lowest_mse:
    col_index_identifier.append(col_index)

# Obtaining the column indices of the top 10 MSE values
top_10_column_indices = [col_index for col_index, _ in top_10_lowest_mse]

# Extract the corresponding top 10 columns from the original dataset
X_top_10 = [[row[col_index] for col_index in top_10_column_indices] for row in fully_cleaned_data_as_float]

# Converting X_top_10 to a NumPy array for further analysis.
X_top_10 = np.array(X_top_10)

# Y does not require any conversion because it is still the last index in each row.
y = [row[-1] for row in fully_cleaned_data_as_float]

# Creating training sets for the final regression model that looks to predict the value for the indicated
# response variable.
X_train, X_test, y_train, y_test = train_test_split(X_top_10, y, test_size=0.2, random_state=42)

# Create a new Random Forest Regressor for the final model
final_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model for the data
final_model.fit(X_train, y_train)

# Predict using the previously trained model
y_prediction = final_model.predict(X_test)

# Calculate R^2 Score.
# The R2 score can be interpreted as the proportion of the variance for a response variable explained by
# the explanatory variables in a regression model.
# A higher value is genuinely desired for R^2 Score because it indicates that the model accounts for more
# variation in the response variable.
r2 = r2_score(y_test, y_prediction)
print(f'R-squared (R2): {r2}')

# Calculate Mean Squared Error (MSE)
# The MSE can be interpreted as the average squared difference between the observed and predicted values.
# A smaller MSE is often desired because that means there is less error, or dispersion from the mean value.
# However, MSE is reliant on the response variable and the context is essential for understanding what is a
# good MSE value.
mse = mean_squared_error(y_test, y_prediction)
print(f'Mean Squared Error (MSE): {mse}')

# Creates a list of the labels that were associated with the top 10 MSE values.
predictor_labels = []
i = 0
for label in clean_column_labels:
    if i in col_index_identifier:
        predictor_labels.append(label)
    i += 1

# Allows the user to input values for each of the top 10 most accurate explanatory variables.
user_input = []
for i in range(10):  # Assuming 10 predictor variables
    value = float(input(f"Enter value for {predictor_labels[i]} ({i + 1}):"))
    user_input.append(value)

# Preprocess user input to match the format used during training.
user_input = np.array(user_input).reshape(1, -1)

# Make predictions using the trained model.
predicted_value = final_model.predict(user_input)

# Displaying the final prediction for the response variable to the user.
print(f"Predicted {Response_variable}: {predicted_value[0]}")

# Next Step in Project: Make a html application where a user can input all the needed stats and the model passes
# the input through to produce the prediction.
