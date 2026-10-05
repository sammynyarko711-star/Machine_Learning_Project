# Explanation Script — Kumasi Sanitation Violation Risk Predictor

Use this as your talking points when walking your lecturer through
`kumasi_sanitation_risk_predictor.ipynb`. It follows the notebook top to
bottom, section by section.

---

## 1. Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

**Say:** "These are the standard Python data science libraries. `numpy` does
numerical operations and random number generation, `pandas` handles the
dataset as a table (a DataFrame), `matplotlib` and `seaborn` create charts,
and the `warnings` lines just silence non-critical warning messages so the
output stays clean."

---

## 2. Creating the synthetic dataset

```python
np.random.seed(42)
n_records = 1000
```

**Say:** "`np.random.seed(42)` fixes the random number generator so that
every time this notebook runs, it produces the *same* random data — this
makes the results reproducible. `n_records = 1000` means we're simulating
1,000 area-records."

```python
sub_metros = ['Asokwa', 'Bantama', 'Suame', 'Manhyia', 'Nhyiaeso', 'Oforikrom', 'Subin', 'Kwadaso']

df = pd.DataFrame({...})
```

**Say:** "I listed 8 real Kumasi sub-metro areas, then built a table where
each row is one 'area-record' with these features:
- `population_density` — random values 500 to 25,000 people per km²
- `distance_to_dumpsite_km` — how far the area is from a waste collection/dump point
- `daily_waste_generated_kg` — estimated daily waste produced
- `number_of_bins` — how many public bins are available
- `road_width_m` — how accessible the area is for collection trucks
- `past_violation_count` — how many past sanitation violations that area has had
- `mainroad_access`, `market_present`, `drainage_present`, `floodprone`, `wastebin_available` — yes/no traits of the area
- `area_type` — residential, commercial, or mixed

Each of these is drawn from a random distribution chosen to look like
realistic Kumasi conditions — for example, `distance_to_dumpsite_km` uses an
exponential distribution because most areas are close to a dump point but a
few are far away, which matches real geography better than a uniform spread."

```python
risk_score = (
    0.00006 * df['population_density']
    + 0.15  * df['distance_to_dumpsite_km']
    + 0.15  * df['past_violation_count']
    - 0.12  * df['number_of_bins']
    - 0.05  * df['road_width_m']
    + df['market_present'].map({'yes': 0.6, 'no': 0.0})
    + df['floodprone'].map({'yes': 0.5, 'no': 0.0})
    - df['drainage_present'].map({'yes': 0.4, 'no': 0.0})
    - df['wastebin_available'].map({'yes': 0.5, 'no': 0.0})
    + df['area_type'].map({'commercial': 0.4, 'mixed': 0.2, 'residential': 0.0})
    - 1.8
)
probability_of_violation = 1 / (1 + np.exp(-risk_score))
df['violation'] = np.random.binomial(1, probability_of_violation)
```

**Say:** "This is the most important part of the data generation — it's how
I encode 'what makes an area risky' into the data, so the dataset has
realistic cause-and-effect patterns rather than being pure noise. Each
feature is given a weight: positive weights (population density, distance,
past violations, market presence, flood-proneness) *increase* risk; negative
weights (more bins, wider roads, drainage present, bins available) *decrease*
risk. I add all these weighted effects together into a `risk_score`.

That score isn't yet a probability — it can be any positive or negative
number. The line `1 / (1 + np.exp(-risk_score))` is the **sigmoid function**,
which squashes any number into a value between 0 and 1 — turning the risk
score into an actual probability. Finally,
`np.random.binomial(1, probability_of_violation)` flips a weighted coin for
each row using that probability, to decide whether a violation actually
happened (1) or not (0) — just like a real-world outcome wouldn't be 100%
predictable even if we know the risk factors."

---

## 3. Initial data inspection

```python
df.head()          # shows the first 5 rows
df.shape           # shows (rows, columns)
df.describe()      # shows count, mean, min, max, etc. for numeric columns
df.info()          # shows column data types and non-null counts
df.loc[df.duplicated()]   # shows any exact duplicate rows
df.columns         # lists all column names
df.isna().sum()    # counts missing values per column
```

**Say:** "This block is standard data-quality checking — the same steps
you'd run on any dataset before modeling. It confirms the data has 1,000
rows, the expected columns, no duplicate rows, and no missing values, since
I generated it cleanly. On real KMA data this step is where you'd actually
catch messy entries, typos, or missing fields."

---

## 4. Exploratory Data Analysis (EDA)

```python
sns.countplot(x='violation', data=df)
```

**Say:** "This plots how many records are violations (1) versus compliant
(0). It shows violations are the minority class — roughly 1 in 5 — which
mirrors reality: most days, most areas are fine."

```python
df['mainroad_access'].value_counts().plot(kind='bar')
```
*(repeated for each yes/no and category column)*

**Say:** "These bar charts just show how the categorical features are
distributed — for example, how many areas have market activity versus not."

```python
sns.barplot(x='market_present', y='violation', data=df)
```

**Say:** "This is the more important chart — it compares the *average
violation rate* between groups. For example, it shows areas with a market
present have a visibly higher violation rate than areas without one, and the
same pattern holds for flood-prone areas. This is a first visual clue that
these features carry real predictive signal."

```python
sns.boxplot(x='violation', y=x, data=df)   # for each numeric feature
```

**Say:** "These boxplots compare the distribution of each numeric feature
between the two groups (violation vs. compliant). Where the boxes are
clearly shifted — like `population_density` and `distance_to_dumpsite_km`
being higher for violations — that's a sign the feature is useful for
prediction. `daily_waste_generated_kg` shows almost total overlap between
the two groups, which is a hint it won't end up significant later."

```python
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
```

**Say:** "This correlation heatmap shows how strongly each numeric feature
relates to every other feature, and to the target. Numbers close to +1 or -1
mean a strong relationship; numbers close to 0 mean little to no linear
relationship."

---

## 5. Encoding categorical variables (dummy variables)

```python
def dummies(x, df):
    temp = pd.get_dummies(df[x], drop_first=True).astype(int)
    temp.columns = [f"{x}_{c}" for c in temp.columns]
    df = pd.concat([df, temp], axis=1)
    df.drop([x], axis=1, inplace=True)
    return df
```

**Say:** "Machine learning models need numbers, not text categories like
'yes'/'no' or 'residential'/'commercial'. This function converts a text
column into 0/1 numeric columns — this is called **one-hot encoding** or
'dummy variables'. `drop_first=True` drops one category to avoid redundancy
(for a yes/no column, if you know it's not 'no', it must be 'yes', so we only
need one column). I applied this function to all six categorical columns."

---

## 6. Train/test split and scaling

```python
df_train, df_test = train_test_split(df_model, train_size=0.75, test_size=0.25,
                                      random_state=100, stratify=df_model['violation'])
```

**Say:** "This splits the data into 75% for training the model and 25% held
back to test it on data it's never seen. `stratify=df_model['violation']`
makes sure both the training and test sets have the same proportion of
violations — important since violations are a minority class."

```python
scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])
```

**Say:** "Our numeric features are on very different scales — population
density goes up to 25,000, while road width only goes up to 12. `MinMaxScaler`
rescales every numeric column to a common 0–1 range, so no single feature
dominates the model just because its numbers happen to be bigger."

```python
y_train = df_train.pop('violation')
X_train = df_train
```

**Say:** "This separates the target we're trying to predict (`y_train` = the
violation column) from the features we're using to predict it (`X_train` =
everything else)."

---

## 7. Feature selection with RFE

```python
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=8)
rfe = rfe.fit(X_train, y_train)
```

**Say:** "RFE stands for Recursive Feature Elimination. Rather than using
every single feature, it repeatedly trains a logistic regression model,
ranks features by importance, removes the weakest one, and repeats — until
only the top 8 remain. This avoids overloading the model with weak or
redundant predictors."

---

## 8. Building and refining the model

```python
def build_model(X, y):
    X = sm.add_constant(X)
    logit_model = sm.Logit(y, X).fit(disp=0)
    print(logit_model.summary())
    return X, logit_model
```

**Say:** "`sm.add_constant(X)` adds an intercept term to the model (a
baseline value when all features are zero). `sm.Logit(y, X).fit()` fits a
**logistic regression** — the standard classification model for a yes/no
outcome — using the `statsmodels` library, which (unlike scikit-learn) gives
us detailed statistics like p-values for each feature, which is what lets us
judge statistical significance."

```python
def checkVIF(X):
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

**Say:** "VIF stands for Variance Inflation Factor. It checks
**multicollinearity** — whether two or more features are so closely related
to each other that the model can't tell which one is actually driving the
prediction. A VIF close to 1 means no problem; VIF above 5 or 10 usually
signals a feature should be dropped."

**Model 1 → Model 2:** "In Model 1, using all 8 RFE-selected features, every
feature had a low VIF (~1.0, no multicollinearity issue at all) but
`daily_waste_generated_kg` had a p-value of 0.195 — well above the 0.05
cutoff for statistical significance — meaning we can't be confident it
actually predicts violations. I dropped it and rebuilt the model as Model 2,
where every remaining feature is statistically significant (p < 0.05). This
mirrors real statistical modeling practice: build a full model, then
simplify by removing what isn't pulling its weight."

---

## 9. Predicting on the test set and evaluating

```python
df_test[numerical_list] = scaler.transform(df_test[numerical_list])
```

**Say:** "Note this uses `.transform()`, not `.fit_transform()` — we reuse
the *exact same scaling* learned from the training data, rather than
re-fitting on the test data. This is crucial: the test set must be treated
exactly like new, unseen data."

```python
y_pred_prob = model_2.predict(X_test_new)
y_pred = (y_pred_prob >= 0.5).astype(int)
```

**Say:** "The model outputs a *probability* between 0 and 1 for each test
row. To get a final yes/no prediction, we apply a 0.5 cutoff — if the
predicted probability is 50% or higher, we call it a predicted violation."

```python
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
classification_report(...)
confusion_matrix(...)
```

**Say:** "`accuracy` is the percentage of correct predictions overall — we
got about 80%. `ROC AUC` measures how well the model ranks violations as
riskier than non-violations across *all* possible cutoffs, not just 0.5 — a
score of 0.5 is random guessing, 1.0 is perfect; we got about 0.71, which is
solidly better than chance. The classification report and confusion matrix
break this down further — showing the model is very good at spotting
compliant areas (97% recall) but misses many actual violations (14% recall),
because violations are the rarer class. This is an honest limitation to
mention, not something to hide."

```python
roc_curve(y_test, y_pred_prob)
```

**Say:** "This plots the ROC curve — how the true positive rate and false
positive rate trade off as we change the cutoff. The further the curve
bulges above the diagonal 'random guess' line, the better the model."

---

## Closing line for your presentation

"This notebook follows a complete, standard machine learning workflow —
data generation, exploration, cleaning, feature selection, iterative model
refinement using p-values and VIF, and honest evaluation on unseen data —
applied to a sanitation risk classification problem for Kumasi. It's
currently trained on synthetic data designed to reflect realistic patterns,
and it's structured so that real KMA data can be substituted in directly
once it's available."
