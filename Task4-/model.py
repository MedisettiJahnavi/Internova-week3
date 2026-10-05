"""Simple linear regression on the diabetes dataset with safe imports

This script is resilient to missing packages and won't error when the
test set is smaller than the hard-coded reporting loop.
"""
from typing import Optional


def main() -> int:
    try:
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
    except Exception as e:  # pragma: no cover - user environment may vary
        print("Missing or failing import:", e)
        print("Please install required packages: pip install scikit-learn")
        return 1

    # Load the dataset
    data = load_diabetes()

    # Prepare X and y
    X = data.data
    y = data.target

    print("Dataset loaded successfully!")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Create the Linear Regression model and train it
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    # Make predictions
    y_pred = model.predict(X_test)

    # Display prediction results (up to available test set size)
    print("\nPrediction Results:")
    n_show = min(10, len(y_test))
    for i in range(n_show):
        print(
            "Actual:", round(float(y_test[i]), 2),
            "| Predicted:", round(float(y_pred[i]), 2)
        )

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Mean Squared Error:", round(mse, 2))
    print("R2 Score:", round(r2, 2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
