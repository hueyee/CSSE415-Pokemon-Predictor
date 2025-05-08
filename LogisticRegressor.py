import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_parquet("Parquets/parsed_showdown_replays.parquet")

X = df.drop(['game_id', 'p1_revealed_pokemon', 'p2_revealed_pokemon'], axis=1)
y = df.p1_revealed_pokemon

X_n = X.select_dtypes(include='number')
X_c = X.select_dtypes(exclude='number')
imputer = SimpleImputer(strategy='mean')
X_processed_imputed = pd.DataFrame(imputer.fit_transform(X_n), columns=X_n.columns)
X_n = StandardScaler().set_output(transform='pandas').fit_transform(X_processed_imputed)
X_c = pd.get_dummies(X_c)
X = pd.concat([X_n,X_c],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 10000]
train_scores = []
test_scores = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_scores.append(train_acc)
    test_scores.append(test_acc)

# Plotting
plt.figure()
plt.plot(C_values, train_scores, marker='o', label='Training Accuracy')
plt.plot(C_values, test_scores, marker='o', label='Testing Accuracy')
plt.xscale('log')  # Because C is best viewed on a log scale
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy')
plt.legend()
plt.grid(True)
plt.show()