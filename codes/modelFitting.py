import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier


def hyperparameter_tuning(X_train_full: pd.DataFrame, y_train_full: pd.Series)-> RandomForestClassifier:
    """
    Performs hyperparameter tuning for a RandomForestClassifier using GridSearchCV.
    
    It then returns a new, final model that has been automatically trained on the entire dataset 
    using the best parameters found.
    
    Parameters:
        X_train_full (pd.DataFrame): The complete training feature set.
        y_train_full (pd.Series): The complete training target labels.

    Returns:
        RandomForestClassifier: The best-performing model, already trained on the full input dataset.
    """
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],      # Number of trees
        'max_depth': [10, 20, 30, None],          # Max depth of each tree
        'min_samples_leaf': [1, 2, 4]         # Min samples at a leaf node
    }

    print("\n--- Starting Hyperparameter Tuning (GridSearch CV) ---")

    # Initialize the base model
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Set up the 5-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up the GridSearch
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_full, y_train_full)

    final_model = grid_search.best_estimator_

    
    # Print the best parameters it found
    print(f"Best parameters found: {grid_search.best_params_}")

    # Print the best average cross-validation score
    print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")

    return final_model



def get_features_importance(final_model: RandomForestClassifier, X_train_full: pd.DataFrame, y_train_full: pd.Series, features: list[str])-> pd.DataFrame:
    """
    Computes, prints, and plots the permutation feature importance for a trained model.

    Parameters:
        final_model: The already trained RandomForest model.
        X_train_full (pd.DataFrame): The complete training feature set.
        y_train_full (pd.Series): The complete training target labels.
        features (list[str]): The list of feature names corresponding to the
                              columns in X_train_full, in the correct order.

    Returns:
        pd.DataFrame: A DataFrame sorted by importance of the feature, containing
                      the importance, feature name, and standard deviation.
    """
    print("\n--- Calculating Feature Importances ---")

    # Run permutation importance on the training data
    result = permutation_importance(
        final_model,
        X_train_full,
        y_train_full,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Put the results into a pandas DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    }).sort_values(by='Importance_Mean', ascending=False)

    print("\nTop 95 Most Important Features:")
    print(importance_df.head(95))

    # Print any features that were useless (0 or negative importance)
    useless_features_count = (importance_df['Importance_Mean'] <= 0).sum()
    print(f"\nFound {useless_features_count} features with 0 or negative importance.")

    top_20 = importance_df.head(20)

    plt.figure(figsize=(10, 12))
    plt.barh(
        top_20['Feature'],
        top_20['Importance_Mean'],
        xerr=top_20['Importance_Std'],
        align='center'
    )
    plt.xlabel("Permutation Importance (Mean Accuracy Drop)")
    plt.title("Top 20 Most Important Features")
    plt.gca().invert_yaxis()  # Display the most important feature at the top
    plt.show()

    return importance_df



def create_lean_model(train_df: pd.DataFrame, test_df: pd.DataFrame, perm_importance_df: pd.DataFrame)-> RandomForestClassifier:
    """
    Performs feature selection and re-trains a new, "lean" model.

    Parameters:
        train_df (pd.DataFrame): The complete training feature set.
        test_df (pd.DataFrame): The complete test feature set.
        perm_importance_df (pd.DataFrame): The DataFrame containing feature names and their 'Importance_Mean'.

    Returns:
        RandomForestClassifier: The new, final, optimized model trained only on the most important features.
    """
    good_features = perm_importance_df[perm_importance_df['Importance_Mean'] > 0]['Feature'].tolist()

    print(f"New 'lean' feature count: {len(good_features)}")

    # Create your new 'lean' training data by selecting only the important features
    X_train_lean = train_df[good_features]
    y_train_lean = train_df['player_won']
    X_submission_test_lean = test_df[good_features]

    # Re-run GridSearch on the LEAN dataset
    print("\n--- Tuning a new model on 'lean' feature set ---")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the base model
    rf_model_lean = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Set up the K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Set up the GridSearch
    lean_grid_search = GridSearchCV(
        estimator=rf_model_lean,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the new, lean model
    lean_grid_search.fit(X_train_lean, y_train_lean)

    # Get the new, final, lean model
    print("\n--- Lean Model Tuning Complete ---")
    print(f"Best parameters: {lean_grid_search.best_params_}")
    print(f"Lean model accuracy: {lean_grid_search.best_score_ * 100:.2f}%")

    final_lean_model = lean_grid_search.best_estimator_

    return final_lean_model, good_features



def create_adaboost_model(X_train_full: pd.DataFrame, y_train_full: pd.Series)-> AdaBoostClassifier:
    """
    Initializes, defines, and trains an AdaBoost classifier.

    Parameters:
        X_train_full (pd.DataFrame): The complete training feature set.
        y_train_full (pd.Series): The complete training target labels.

    Returns:
        AdaBoostClassifier: The trained AdaBoost model.
    """

    # Define the base estimator
    base_estimator = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    # Define the Adaboost model
    adaboost_model = AdaBoostClassifier(
        estimator=base_estimator, 
        n_estimators=500,        
        learning_rate=0.1,       
        random_state=42
    )

    print("Training of AdaBoost model...")
    adaboost_model.fit(X_train_full, y_train_full)
    print("Training complete!!")

    return adaboost_model



def create_voting_ensemble_model(final_model: RandomForestClassifier, adaboost_model: AdaBoostClassifier, X_train_full: pd.DataFrame, y_train_full: pd.Series)-> VotingClassifier:
    """
    Creates and trains a 'soft' voting ensemble model.

    Parameters:
        final_model (RandomForestClassifier): An already trained RandomForest model.
        adaboost_model (AdaBoostClassifier): An already trained AdaBoost model.
        X_train_full (pd.DataFrame): The complete training feature set.
        y_train_full (pd.Series): The complete training target labels.

    Returns:
        VotingClassifier: The trained 'soft' voting ensemble model.
    """

    # Define the voting ensemble by taking two models
    voting_ensemble = VotingClassifier(
        estimators=[
            ('rf', final_model),      
            ('ab', adaboost_model)    
        ],
        voting='soft', 
        n_jobs=-1     
    )

    # Train the model on the training set
    print("Training of Voting Ensemble model...")
    voting_ensemble.fit(X_train_full, y_train_full) 
    print("Training complete!!")

    return voting_ensemble