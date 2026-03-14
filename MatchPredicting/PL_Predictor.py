## Erik Cupsa
## PL Predictor using scikit-learn to predict from matches.csv stat sheet

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load dataset
matches = pd.read_csv(
    r"C:\Users\hp\Downloads\PLWebsite-main\MatchPredicting\matches.csv",
    index_col=0
)

# Convert data types
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes
matches["opp"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
matches["day"] = matches["date"].dt.dayofweek

# Target variable (Win = 1)
matches["target"] = (matches["result"] == "W").astype(int)

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=10,
    random_state=1
)

# Train/test split
train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] > "2022-01-01"]

predictors = ["h/a", "opp", "hour", "day"]

rf.fit(train[predictors], train["target"])

# Predictions
preds = rf.predict(test[predictors])

# Accuracy
acc = accuracy_score(test["target"], preds)
print("Accuracy:", acc)

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
print(pd.crosstab(index=combined["actual"], columns=combined["prediction"]))

print("Precision:", precision_score(test["target"], preds))


# -------- Rolling Average Feature Engineering --------

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester United").sort_values("date")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")

    rolling_stats = group[cols].rolling(3, closed="left").mean()

    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)

    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

# Apply rolling averages to all teams
matches_rolling = matches.groupby("team", group_keys=False).apply(
    rolling_averages, cols=cols, new_cols=new_cols
)

matches_rolling.index = range(matches_rolling.shape[0])


# -------- Model with Rolling Features --------

def make_predictions(data, predictors):

    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]

    rf.fit(train[predictors], train["target"])

    preds = rf.predict(test[predictors])

    combined = pd.DataFrame(
        dict(actual=test["target"], prediction=preds),
        index=test.index
    )

    precision = precision_score(test["target"], preds)

    return combined, precision


combined, precision = make_predictions(matches_rolling, predictors + new_cols)

print("Precision with rolling features:", precision)

combined = combined.merge(
    matches_rolling[["date", "team", "opponent", "result"]],
    left_index=True,
    right_index=True
)

print(combined.head())


# -------- Team Name Fix Mapping --------

class MissingDict(dict):
    __missing__ = lambda self, key: key


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

# Merge predictions for both teams in same match
merged = combined.merge(
    combined,
    left_on=["date", "new_team"],
    right_on=["date", "opponent"]
)

print(merged.head())


## Project inspired by Dataquest tutorial