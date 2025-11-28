"""
Business Prediction - Main Application
Uses modular components from modules/ directory
"""
import warnings

from prediction_modules.data_loader import load_all_data
from prediction_modules.preprocessing import preprocess_data
from prediction_modules.model_training import train_and_evaluate_pipeline
from prediction_modules.predictions import run_predictions

warnings.simplefilter("ignore")


def main():
    """Main application pipeline."""

    # Step 1: Load data
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    cafe_df, population_df = load_all_data()

    print(f"\n{population_df.head()=}")
    print(f"\n{cafe_df.head()=}")
    print(f"\n{cafe_df.info()=}")
    print(f"\nCafe data shape: {cafe_df.shape}")
    print(f"Population data shape: {population_df.shape}")
    print(f"\n{cafe_df.describe()=}")

    # Optional: Show some charts
    ax = cafe_df['City'].value_counts().head(5).plot(kind='bar')
    ax.set_title('Top 5 Cities with most Coffee Shops')
    # plt.show()

    ax = cafe_df['Business Name'].value_counts().head(10).plot(kind='bar')
    ax.set_title('Top 10 most famous brands')
    # plt.show()

    print(f"\nMissing values:\n{cafe_df.isna().sum()}")

    # Step 2: Preprocess data
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING DATA")
    print("="*60)
    processed_df, top_zip_codes_df = preprocess_data(cafe_df, population_df)

    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"\nProcessed data:\n{processed_df.head()}")
    print(f"\nTop 5 zip codes:\n{top_zip_codes_df.head()}")
    print(f"Top Zip Codes: {top_zip_codes_df['Zip Code'].unique().tolist()}")

    # Step 3: Train and evaluate models
    print("\n" + "="*60)
    print("STEP 3: TRAINING AND EVALUATING MODELS")
    print("="*60)
    trained_models, scaler, results = train_and_evaluate_pipeline(processed_df, tune=True)

    print(f"\n✓ Trained {len(trained_models)} models")

    # Step 4: Make predictions
    print("\n" + "="*60)
    print("STEP 4: MAKING PREDICTIONS")
    print("="*60)
    predictions_df, price_range_df = run_predictions(trained_models, scaler, top_zip_codes_df)

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS: Price Range by Zip Code")
    print("="*60)
    print(price_range_df.to_string(index=False))

    print("\n" + "="*60)
    print("FINAL RESULTS: All Model Predictions")
    print("="*60)
    print(predictions_df.to_string(index=False))

    print("\n✓ Pipeline completed successfully!")

    return predictions_df, price_range_df, trained_models, scaler


if __name__ == "__main__":
    predictions_df, price_range_df, models, scaler = main()
