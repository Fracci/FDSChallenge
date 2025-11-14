# Kaggle Competition - Predicting Pokèmon battle Outcomes
**Course of Foundations of Data Science**

## Introduction
This report explains the methodology we used for a bi-
nary classification challenge that consisted in predicting the
winner of a Pokèmon battle (Player 1 or Player 2) using
only traditional Machine Learning models. The informa-
tion given were the composition of Player 1’s team, Player
2’s starting Pokèmon and the first 30 turns of the battle.

## Method
The methodology that lead us to the prediction was divided
into three main phases: Feature Engineering, Model Vali-
dation and Feature Selection.

### Feature Engineering
Data were provided in a complex, nested JSON fromat, ne-
cessitating of a robust pipeline to convert them into a high-
dimensional feature vector. We engineered three different
classes of features to capture both the pre-battle advantages
and the strategies applied during the battle timeline.
* **Static Features** that don’t change during the battle be-
cause they are related to the teams’ composition (e.g.,
their stats, types, matchups). The Pokèmons of Player 2’s
team are extracted from the battle timeline (the number of
Pokèmons he reveals becomes an essential feature).
* **Dynamic Features**: we used a Pokèmon state tracker to
create features that are useful to understand the strategies
of the players during the single turns of the battle (by the
number of switches, and the offense and defense power
they have).
* **Last Turn Features**: A ”snapshot” of the last turn, to un-
derstand which player has an advantage in the total HP
left, in the number of Pokèmons left and in the active
Pokèmon battle during turn 30.

### Model Selection and Validation
Once we had all the flat set of features, we could start ex-
perimenting by fitting different Machine Learning models.
We tested three models:
* **Random Forest Classifier**
* **AdaBoost Classifier**
* **Voting Ensemble** (of the previous two models)

To select the best model and parameters, we used **K-Fold
cross-validation** (with K=5), that provides a robust and re-
liable measure of model performance, and avoids overfitting
by training and evaluating the model on 5 different splits of
the data. This validation process showed that **Random Forest Classifier** outperformed the other two models, yielding
a higher average accuracy. We also thought that Logistic
Regression could be a good fit but it wasn’t the right fit for
our features (that are highly correlated and not normalized).
We then performed hyperparameter tuning by using
GridSearchCV to automatically test and tune the Random
Foresy hyperparameters. This ensured that our final model
was not only well-validated but also optimally tuned for the
dataset.

### Feature Selection and Final Model
After training an initial Random Forest on all 93 features,
we analyzed it using **Permutation Importance**, that com-
putes each feature’s importance by measuring the drop in
accuracy when it is random shuffled. Thus, we could have
an idea of which were the most important features to create
a ”lean model” by removing the features (like priority move
count) that had a 0 or negative importance since they simply
represent random noise and have no utility in the model and
were probably ignored by most trees. We then re-ran Grid-
SearchCV on this new, smaller feature set. This process re-
sulted in a final model that was faster to train and less prone
to overfitting, which obtained an accuracy of **84.86%**.

## Conclusion
The key to this challenge was not complex model archi-
tecture but exhaustive feature engineering. By successfully
translating the turn-by-turn state of the battle into a rich set
of numerical features, we gave a Random Forest all the in-
formation it needed to find the underlying patterns of vic-
tory. It proved to be the best model because it is able to
handle high-dimensional, non-linear data, its resistance to
overfitting and it doesn’t require feature scaling.



_By Francesco Finazzi, Chiara Vulpiani and Relja Savic_
