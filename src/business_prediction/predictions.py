"""Module for making predictions with trained models."""
import pandas as pd


def predict_for_zip_codes(models, scaler, top_zip_codes_df, exclude_columns=None):
    """
    Make predictions for top zip codes using all models.

    Args:
        models: Dictionary of trained models
        scaler: Fitted StandardScaler
        top_zip_codes_df: DataFrame with top zip codes data
        exclude_columns: Columns to exclude (default: ['Zip Code', 'Latte Price'])

    Returns:
        DataFrame with predictions from all models
    """
    if exclude_columns is None:
        exclude_columns = ['Zip Code', 'Latte Price']

    # Prepare features
    zip_codes_features = top_zip_codes_df.drop(exclude_columns, axis=1)
    zip_codes_scaled = scaler.transform(zip_codes_features)

    # Make predictions with each model
    predictions = {}
    for model_name, model in models.items():
        predicted_prices = model.predict(zip_codes_scaled)
        predictions[model_name] = predicted_prices

        print(f"\nPredicted Latte Prices by {model_name} for Top 5 Zip Codes:")
        print(predicted_prices)

    # Create predictions dataframe
    predictions_df = pd.DataFrame(predictions)
    predictions_df['Zip Code'] = top_zip_codes_df['Zip Code'].values

    # Reorder columns to put Zip Code first
    cols = ['Zip Code'] + [col for col in predictions_df.columns if col != 'Zip Code']
    predictions_df = predictions_df[cols]

    return predictions_df


def get_price_range_by_zip(predictions_df, model_name='Linear Regression'):
    """
    Get highest and lowest predicted prices by zip code.

    Args:
        predictions_df: DataFrame with predictions
        model_name: Model to use for price range (default: 'Linear Regression')

    Returns:
        DataFrame with highest and lowest prices per zip code
    """
    agg_df = predictions_df.groupby('Zip Code')[model_name].agg([
        ("Highest", "max"),
        ("Lowest", "min")
    ]).reset_index()

    agg_df.columns = ['Zip Code', 'Highest Price', 'Lowest Price']

    return agg_df


def run_predictions(models, scaler, top_zip_codes_df):
    """
    Complete prediction pipeline.

    Args:
        models: Dictionary of trained models
        scaler: Fitted StandardScaler
        top_zip_codes_df: DataFrame with top zip codes

    Returns:
        Tuple of (predictions_df, price_range_df)
    """
    # Make predictions
    predictions_df = predict_for_zip_codes(models, scaler, top_zip_codes_df)

    # Get price ranges
    price_range_df = get_price_range_by_zip(predictions_df)

    return predictions_df, price_range_df
