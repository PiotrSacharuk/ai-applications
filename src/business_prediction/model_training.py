"""Module for model training, hyperparameter tuning, and evaluation."""
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter("ignore")


def prepare_train_test_data(df, target_column='Latte Price', exclude_columns=None, test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets.

    Args:
        df: Input DataFrame
        target_column: Name of the target column (default: 'Latte Price')
        exclude_columns: List of columns to exclude from features (default: ['Latte Price', 'Zip Code'])
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed (default: 42)

    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test)
    """
    if exclude_columns is None:
        exclude_columns = [target_column, 'Zip Code']

    X = df.drop(exclude_columns, axis=1)
    Y = df[target_column]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, Y_train, Y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def get_models():
    """
    Get dictionary of models to train.

    Returns:
        Dictionary of model_name: model_instance
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor()
    }


def get_param_grid():
    """
    Get hyperparameter grid for model tuning.

    Returns:
        Dictionary of model_name: param_grid
    """
    return {
        'Random Forest Regressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        },
        'Gradient Boosting Regressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
    }


def tune_hyperparameters(models, X, Y, param_grid=None, cv=5):
    """
    Tune hyperparameters using GridSearchCV.

    Args:
        models: Dictionary of models
        X: Feature matrix
        Y: Target vector
        param_grid: Parameter grid for tuning (default: None, uses default grid)
        cv: Number of cross-validation folds (default: 5)

    Returns:
        Dictionary of tuned models
    """
    if param_grid is None:
        param_grid = get_param_grid()

    tuned_models = models.copy()

    for model_name, model in models.items():
        if model_name in param_grid:
            print(f"Tuning {model_name}...")
            grid_search = GridSearchCV(
                model,
                param_grid[model_name],
                cv=cv,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X, Y)
            tuned_models[model_name] = grid_search.best_estimator_
            print(f"Best params for {model_name}: {grid_search.best_params_}")

    return tuned_models


def train_models(models, X_train, Y_train):
    """
    Train all models.

    Args:
        models: Dictionary of models
        X_train: Training features
        Y_train: Training target

    Returns:
        Dictionary of trained models
    """
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, Y_train)
        trained_models[model_name] = model

    return trained_models


def evaluate_models(models, X_test, Y_test):
    """
    Evaluate all models and print metrics.

    Args:
        models: Dictionary of trained models
        X_test: Testing features
        Y_test: Testing target

    Returns:
        Dictionary of model_name: metrics_dict
    """
    results = {}

    for model_name, model in models.items():
        Y_pred = model.predict(X_test)

        metrics = {
            'MAE': mean_absolute_error(Y_test, Y_pred),
            'MSE': mean_squared_error(Y_test, Y_pred),
            'R2': r2_score(Y_test, Y_pred)
        }

        results[model_name] = metrics

        print(f"\n{model_name} Metrics:")
        print(f"Mean Absolute Error: {metrics['MAE']:.4f}")
        print(f"Mean Squared Error: {metrics['MSE']:.4f}")
        print(f"R^2 Score: {metrics['R2']:.4f}")

    return results


def train_and_evaluate_pipeline(df, tune=True):
    """
    Complete training and evaluation pipeline.

    Args:
        df: Preprocessed DataFrame
        tune: Whether to perform hyperparameter tuning (default: True)

    Returns:
        Tuple of (trained_models, scaler, evaluation_results)
    """
    # Prepare data
    X_train, X_test, Y_train, Y_test = prepare_train_test_data(df)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Get models
    models = get_models()

    # Tune hyperparameters if requested
    if tune:
        X = df.drop(['Latte Price', 'Zip Code'], axis=1)
        Y = df['Latte Price']
        X_scaled = scaler.fit_transform(X)
        models = tune_hyperparameters(models, X_scaled, Y)

    # Train models
    trained_models = train_models(models, X_train_scaled, Y_train)

    # Evaluate models
    results = evaluate_models(trained_models, X_test_scaled, Y_test)

    return trained_models, scaler, results
