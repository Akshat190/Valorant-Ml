"""
Main script for Valorant VCT Winner Prediction System
"""

import os
import sys
import pandas as pd
from datetime import datetime
import argparse
from typing import Dict, List

# Import our modules
from data_collector import ValorantDataCollector
from data_preprocessor import ValorantDataPreprocessor
from feature_engineering import ValorantFeatureEngineer
from ml_model import ValorantMLModel
from evaluation import ValorantModelEvaluator
from prediction_interface import ValorantPredictionInterface


def collect_data():
    """Collect data from Valorant Esports APIs"""
    print("=" * 60)
    print("COLLECTING DATA FROM VALORANT ESPORTS APIs")
    print("=" * 60)
    
    collector = ValorantDataCollector()
    raw_data = collector.collect_all_data(fetch_all_events=True, fetch_all_matches=True)
    
    # Save raw data (JSON + CSV)
    collector.save_data(raw_data)
    
    # Additionally save events and matches with special formatting
    if raw_data['events']:
        events_df = collector.save_events_to_csv(raw_data['events'], "valorant_events_detailed.csv")
        print(f"Events CSV saved with {len(events_df)} events")
    
    if raw_data['matches']:
        matches_df = collector.save_matches_to_csv(raw_data['matches'], "valorant_matches_detailed.csv")
        print(f"Matches CSV saved with {len(matches_df)} matches")
    
    print(f"Data collection completed!")
    print(f"Results: {len(raw_data['results'])} matches")
    print(f"Teams: {len(raw_data['teams'])} teams")
    print(f"Players: {len(raw_data['players'])} players")
    print(f"Events: {len(raw_data['events'])} events")
    print(f"Matches: {len(raw_data['matches'])} matches")
    
    return raw_data


def preprocess_data(raw_data: Dict):
    """Preprocess collected data"""
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    preprocessor = ValorantDataPreprocessor()
    df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
    
    # Save processed data
    preprocessor.save_processed_data(df, team_stats, region_strength, h2h_stats)
    
    print(f"Data preprocessing completed!")
    print(f"Processed matches: {len(df)}")
    print(f"Teams with stats: {len(team_stats)}")
    print(f"Regions analyzed: {len(region_strength)}")
    print(f"H2H pairs: {len(h2h_stats)}")
    
    return df, team_stats, region_strength, h2h_stats


def engineer_features(df: pd.DataFrame, team_stats: Dict, region_strength: Dict, h2h_stats: Dict):
    """Engineer features for ML model"""
    print("\n" + "=" * 60)
    print("ENGINEERING FEATURES")
    print("=" * 60)
    
    feature_engineer = ValorantFeatureEngineer(team_stats, region_strength, h2h_stats)
    X, y, metadata, feature_names = feature_engineer.prepare_ml_features(df)
    
    print(f"Feature engineering completed!")
    print(f"Features created: {len(feature_names)}")
    print(f"Training samples: {len(X)}")
    print(f"Feature names: {feature_names}")
    
    return X, y, metadata, feature_names


def train_model(X, y, feature_names):
    """Train the ML model"""
    print("\n" + "=" * 60)
    print("TRAINING MACHINE LEARNING MODEL")
    print("=" * 60)
    
    ml_model = ValorantMLModel()
    results = ml_model.train_full_pipeline(X, y, feature_names)
    
    print(f"Model training completed!")
    print(f"Best model: {results['best_model']}")
    print(f"Final accuracy: {results['accuracy']:.4f}")
    print(f"Final AUC: {results['auc']:.4f}")
    
    return ml_model, results


def evaluate_model(ml_model, X, y, feature_names):
    """Evaluate the trained model"""
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from config import MODEL_CONFIG
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG['test_size'], 
        random_state=MODEL_CONFIG['random_state'],
        stratify=y
    )
    
    # Scale features
    X_train_scaled = ml_model.scaler.transform(X_train)
    X_test_scaled = ml_model.scaler.transform(X_test)
    
    # Create evaluator
    evaluator = ValorantModelEvaluator(ml_model.best_model, ml_model.scaler, feature_names)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(X_test_scaled, y_test, X_train_scaled, y_train)
    print(report)
    
    # Save evaluation plots
    evaluator.save_evaluation_plots(X_test_scaled, y_test, X_train_scaled, y_train)
    
    return evaluator


def create_prediction_interface(ml_model, team_stats, region_strength, h2h_stats):
    """Create prediction interface"""
    print("\n" + "=" * 60)
    print("CREATING PREDICTION INTERFACE")
    print("=" * 60)
    
    predictor = ValorantPredictionInterface()
    predictor.model = ml_model.best_model
    predictor.scaler = ml_model.scaler
    predictor.feature_names = ml_model.feature_names
    predictor.team_stats = team_stats
    predictor.region_strength = region_strength
    predictor.h2h_stats = h2h_stats
    
    print("Prediction interface created!")
    print("You can now make predictions using the predictor object.")
    
    return predictor


def demo_predictions(predictor):
    """Demonstrate prediction capabilities"""
    print("\n" + "=" * 60)
    print("DEMONSTRATION PREDICTIONS")
    print("=" * 60)
    
    # Example predictions (these would work with real data)
    example_matches = [
        {"team1_name": "G2 Esports", "team2_name": "Team Heretics", 
         "team1_country": "us", "team2_country": "eu"},
        {"team1_name": "Sentinels", "team2_name": "NRG", 
         "team1_country": "us", "team2_country": "us"},
        {"team1_name": "DRX", "team2_name": "T1", 
         "team1_country": "kr", "team2_country": "kr"}
    ]
    
    print("Example match predictions:")
    for match in example_matches:
        try:
            prediction = predictor.predict_match(
                match["team1_name"], 
                match["team2_name"],
                match.get("team1_country"),
                match.get("team2_country")
            )
            print(f"\n{match['team1_name']} vs {match['team2_name']}:")
            print(f"  Predicted Winner: {prediction['predicted_winner']}")
            print(f"  Confidence: {prediction['confidence']:.2%}")
            print(f"  Team 1 Win Probability: {prediction['team1_win_probability']:.2%}")
            print(f"  Team 2 Win Probability: {prediction['team2_win_probability']:.2%}")
        except Exception as e:
            print(f"Error predicting {match}: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Valorant VCT Winner Prediction System')
    parser.add_argument('--mode', choices=['full', 'collect', 'collect-csv', 'collect-all-regions', 'preprocess', 'train', 'predict', 'analyze-csv'], 
                       default='full', help='Execution mode')
    parser.add_argument('--model-path', type=str, help='Path to saved model for prediction mode')
    parser.add_argument('--team1', type=str, help='Team 1 name for prediction')
    parser.add_argument('--team2', type=str, help='Team 2 name for prediction')
    parser.add_argument('--country1', type=str, help='Team 1 country for prediction')
    parser.add_argument('--country2', type=str, help='Team 2 country for prediction')
    parser.add_argument('--status', choices=['all', 'ongoing', 'upcoming', 'completed'], 
                       default='all', help='Filter events by status (for collect-csv mode)')
    parser.add_argument('--region', choices=['all', 'na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc'],
                       default='all', help='Filter events by region (for collect-csv mode)')
    
    args = parser.parse_args()
    
    print("VALORANT VCT WINNER PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Execution mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.mode == 'full':
        # Complete pipeline
        raw_data = collect_data()
        df, team_stats, region_strength, h2h_stats = preprocess_data(raw_data)
        X, y, metadata, feature_names = engineer_features(df, team_stats, region_strength, h2h_stats)
        ml_model, results = train_model(X, y, feature_names)
        evaluator = evaluate_model(ml_model, X, y, feature_names)
        predictor = create_prediction_interface(ml_model, team_stats, region_strength, h2h_stats)
        demo_predictions(predictor)
        
    elif args.mode == 'collect':
        collect_data()
        
    elif args.mode == 'collect-csv':
        # Collect data and save to CSV with filtering
        from collect_events_matches import collect_all_data_with_csv
        data = collect_all_data_with_csv()
        
    elif args.mode == 'collect-all-regions':
        # Collect data from all VCT regions
        from collect_all_regions import collect_all_regions_data
        all_data, summaries = collect_all_regions_data(args.status, True)
        print(f"\nCollected data from {len(summaries)} regions")
        
    elif args.mode == 'analyze-csv':
        # Analyze existing CSV data
        from analyze_csv_data import main as analyze_main
        analyze_main()
        
    elif args.mode == 'preprocess':
        # Load existing raw data
        print("Loading existing raw data...")
        # This would load from saved files
        print("Preprocessing mode requires raw data files.")
        
    elif args.mode == 'train':
        print("Training mode requires preprocessed data files.")
        
    elif args.mode == 'predict':
        if not args.model_path or not args.team1 or not args.team2:
            print("Prediction mode requires --model-path, --team1, and --team2 arguments")
            return
        
        # Load model and make prediction
        predictor = ValorantPredictionInterface(args.model_path)
        prediction = predictor.predict_match(args.team1, args.team2, args.country1, args.country2)
        
        print(f"\nPREDICTION RESULT:")
        print(f"Match: {args.team1} vs {args.team2}")
        print(f"Predicted Winner: {prediction['predicted_winner']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Team 1 Win Probability: {prediction['team1_win_probability']:.2%}")
        print(f"Team 2 Win Probability: {prediction['team2_win_probability']:.2%}")
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
