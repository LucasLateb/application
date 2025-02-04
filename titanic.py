"""
This module performs data loading, exploratory analysis, and
training of a Random Forest classifier on the Titanic dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

# os.chdir('/home/onyxia/work/application')
TrainingData = pd.read_csv("data.csv")

TrainingData.head()


TrainingData["Ticket"].str.split("/").str.len()

TrainingData["Name"].str.split(",").str.len()

parser = argparse.ArgumentParser(description="Combien d'arbres le random forest doit contenir?")
parser.add_argument("--N_TREES", type=int, help="Nombre d'arbres du modèle")
args = parser.parse_args()

# Si l’utilisateur n’a pas fourni l’argument --N_TREES
# on lui pose la question directement
if args.N_TREES is None:
    saisie = input(f"{parser.description}\n> ")  # On affiche la description comme question
    # Si l’utilisateur n’a rien saisi, on peut mettre une valeur par défaut
    if not saisie.strip():
        saisie = "20"
    args.N_TREES = int(saisie)

print(f"Valeur de N_TREES : {args.N_TREES}")


# N_TREES = 20
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

TrainingData.isnull().sum()


## Un peu d'exploration et de feature engineering

### Statut socioéconomique

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age

sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie
# Pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train]).to_csv("train.csv")
pd.concat([X_test, y_test]).to_csv("test.csv")

JETONAPI = "$trotskitueleski1917"


# Random Forest
# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
