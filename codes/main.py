import json
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

from lastTurnFeatures import create_features
from modelFitting import create_lean_model, get_features_importance, create_adaboost_model, create_voting_ensemble_model
pd.set_option('display.max_rows', None) # Used to show all the features


competition = 'fds-pokemon-battles-prediction-2025'
data_path = os.path.join('../input', competition)

train_file_path = os.path.join(data_path, 'train.jsonl')
test_file_path = os.path.join(data_path, 'test.jsonl')
train_data = []

# Read the file line by line
print(f"Loading data from '{train_file_path}'...")
try:
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
            # Append every line of the json file to the train_data file

    print(f"Successfully loaded {len(train_data)} battles.")

    # Let's print the first battle to see its structure 
    print("\n--- Structure of the first train battle: ---")
    if train_data:
        first_battle = train_data[0]
        
        # We create a copy and truncate the battle timeline to keep the output clean
        battle_for_display = first_battle.copy()
        #battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2] # Show only first 2 turns
        
        # Use json.dumps for pretty-printing the dictionary
        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")

# Handle the error if the file is not found at the specified path
except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{train_file_path}'.")
    print("Please make sure you have added the competition data to this notebook.")



# Create feature DataFrames for both training and test sets
print("Processing training data...")
train_df = create_features(train_data)

print("\nProcessing test data...")
test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
test_df = create_features(test_data)

#print("\nTraining features preview:")
#print(train_df.iloc[0])
#display(train_df[:20])



# Define the full training dataset
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X_train_full = train_df[features]
y_train_full = train_df['player_won']

# Define the test dataset
X_submission_test = test_df[features] 

print(f"Total training samples: {len(X_train_full)}")
print(f"Total features: {len(features)}")

# We commented out this line because we already got the best hyperparameters
#final_model = hyperparameter_tuning(X_train_full, y_train_full)

print("\n--- Tuning Complete ---")


final_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=30,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1        
)

# Train the model on ALL training data
print("Training the final model with best parameters...")
final_model.fit(X_train_full, y_train_full)
print("Final model trained.")

# We commented out these two lines because we tried to use the models but they yielded a worse accuracy than RandomForest
#adaboost_model = create_adaboost_model(X_train_full, y_train_full)
#voting_ensemble_model = create_voting_ensemble_model(final_model, adaboost_model, X_train_full, y_train_full)

print("Selecting the best features...")

# Create the lean model using only the most important features and fit it
perm_importance_df = get_features_importance(final_model, X_train_full, y_train_full, features)

tuple1 = create_lean_model(train_df, test_df, perm_importance_df)
final_lean_model = tuple1[0]
good_features = tuple1[1]

print("\nFinal model (with best parameters) is ready.")



print("\n--- Making Final Predictions on Test Data ---")

# Use the final model to predict the class (0 or 1)
X_submission_test_lean = test_df[good_features]
submission_predictions = final_lean_model.predict(X_submission_test_lean)

print("Predictions generated successfully.")

# Create a final DataFrame with the battle IDs and predictions
submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': submission_predictions,
})

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully.")
print(submission_df.head())