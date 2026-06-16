import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report 
)
  

def main():

    df = pd.read_csv("data/processed/wdbc_data_normalized.csv")

    # Converter classes
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Cross-validation

    scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

    print("\n===== LOGISTIC REGRESSION =====")

    print("\n===== CROSS VALIDATION =====")
    print("Scores:", scores)
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")

    #Treinar o modelo

    model.fit(X_train, y_train)

    # Fazer previsões

    y_pred = model.predict(X_test)

    

    # Metricas de avaliação

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

    importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})

    importance["abs_coef"] = importance["coefficient"].abs()

    importance = importance.sort_values(
        by="abs_coef",
        ascending=False
    )

    print("\nTop 10 Features:")
    print(
        importance[
            ["feature", "coefficient"]
        ].head(10)
    )

    # Matriz de confusão

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # Relatório de classificação

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()