"""
Demo script for Valorant VCT Winner Prediction System
This script demonstrates the complete pipeline with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def create_sample_data():
    """Create sample data for demonstration purposes"""
    print("Creating sample data for demonstration...")
    
    # Sample match results
    sample_results = [
        {
            "id": "1",
            "teams": [
                {"name": "G2 Esports", "score": "2", "country": "us", "won": True},
                {"name": "Team Heretics", "score": "0", "country": "eu", "won": False}
            ],
            "status": "Completed",
            "ago": "2h 15m",
            "event": "Group Stage–Opening (D)",
            "tournament": "Valorant Champions 2025"
        },
        {
            "id": "2",
            "teams": [
                {"name": "Sentinels", "score": "1", "country": "us", "won": False},
                {"name": "NRG", "score": "2", "country": "us", "won": True}
            ],
            "status": "Completed",
            "ago": "5h 30m",
            "event": "Group Stage–Opening (C)",
            "tournament": "Valorant Champions 2025"
        },
        {
            "id": "3",
            "teams": [
                {"name": "DRX", "score": "2", "country": "kr", "won": True},
                {"name": "T1", "score": "0", "country": "kr", "won": False}
            ],
            "status": "Completed",
            "ago": "1d 2h",
            "event": "Group Stage–Opening (B)",
            "tournament": "Valorant Champions 2025"
        },
        {
            "id": "4",
            "teams": [
                {"name": "Team Liquid", "score": "0", "country": "eu", "won": False},
                {"name": "GIANTX", "score": "2", "country": "eu", "won": True}
            ],
            "status": "Completed",
            "ago": "1d 5h",
            "event": "Group Stage–Opening (A)",
            "tournament": "Valorant Champions 2025"
        },
        {
            "id": "5",
            "teams": [
                {"name": "Paper Rex", "score": "2", "country": "sg", "won": True},
                {"name": "EDward Gaming", "score": "1", "country": "cn", "won": False}
            ],
            "status": "Completed",
            "ago": "2d 3h",
            "event": "Group Stage–Opening (A)",
            "tournament": "Valorant Champions 2025"
        }
    ]
    
    # Add more historical data for better statistics
    historical_matches = []
    teams = ["G2 Esports", "Team Heretics", "Sentinels", "NRG", "DRX", "T1", "Team Liquid", "GIANTX", "Paper Rex", "EDward Gaming"]
    countries = ["us", "eu", "us", "us", "kr", "kr", "eu", "eu", "sg", "cn"]
    
    for i in range(50):  # Create 50 historical matches
        team1_idx = np.random.randint(0, len(teams))
        team2_idx = np.random.randint(0, len(teams))
        while team2_idx == team1_idx:
            team2_idx = np.random.randint(0, len(teams))
        
        team1_wins = np.random.random() > 0.5
        team1_score = np.random.randint(0, 3) if team1_wins else np.random.randint(0, 2)
        team2_score = 2 - team1_score if team1_wins else np.random.randint(1, 3)
        
        historical_matches.append({
            "id": str(100 + i),
            "teams": [
                {"name": teams[team1_idx], "score": str(team1_score), "country": countries[team1_idx], "won": team1_wins},
                {"name": teams[team2_idx], "score": str(team2_score), "country": countries[team2_idx], "won": not team1_wins}
            ],
            "status": "Completed",
            "ago": f"{np.random.randint(1, 30)}d {np.random.randint(1, 24)}h",
            "event": "Group Stage",
            "tournament": "Valorant Champions 2025"
        })
    
    all_results = sample_results + historical_matches
    
    return {
        'results': all_results,
        'teams': [],
        'players': [],
        'events': [],
        'matches': []
    }

def run_demo():
    """Run the complete demo pipeline"""
    print("=" * 60)
    print("VALORANT VCT WINNER PREDICTION SYSTEM - DEMO")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        print("\n1. Creating sample data...")
        raw_data = create_sample_data()
        print(f"   ✓ Created {len(raw_data['results'])} sample matches")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        from data_preprocessor import ValorantDataPreprocessor
        preprocessor = ValorantDataPreprocessor()
        df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
        print(f"   ✓ Processed {len(df)} matches")
        print(f"   ✓ Calculated stats for {len(team_stats)} teams")
        print(f"   ✓ Analyzed {len(region_strength)} regions")
        
        # Step 3: Engineer features
        print("\n3. Engineering features...")
        from feature_engineering import ValorantFeatureEngineer
        feature_engineer = ValorantFeatureEngineer(team_stats, region_strength, h2h_stats)
        X, y, metadata, feature_names = feature_engineer.prepare_ml_features(df)
        print(f"   ✓ Created {len(feature_names)} features")
        print(f"   ✓ Generated {len(X)} training samples")
        
        # Step 4: Train model
        print("\n4. Training machine learning model...")
        from ml_model import ValorantMLModel
        ml_model = ValorantMLModel()
        results = ml_model.train_full_pipeline(X, y, feature_names)
        print(f"   ✓ Best model: {results['best_model']}")
        print(f"   ✓ Accuracy: {results['accuracy']:.2%}")
        print(f"   ✓ AUC Score: {results['auc']:.4f}")
        
        # Step 5: Create prediction interface
        print("\n5. Creating prediction interface...")
        from prediction_interface import ValorantPredictionInterface
        predictor = ValorantPredictionInterface()
        predictor.model = ml_model.best_model
        predictor.scaler = ml_model.scaler
        predictor.feature_names = ml_model.feature_names
        predictor.team_stats = team_stats
        predictor.region_strength = region_strength
        predictor.h2h_stats = h2h_stats
        print("   ✓ Prediction interface ready")
        
        # Step 6: Make sample predictions
        print("\n6. Making sample predictions...")
        sample_matches = [
            {"team1_name": "G2 Esports", "team2_name": "Team Heretics", "team1_country": "us", "team2_country": "eu"},
            {"team1_name": "Sentinels", "team2_name": "NRG", "team1_country": "us", "team2_country": "us"},
            {"team1_name": "DRX", "team2_name": "T1", "team1_country": "kr", "team2_country": "kr"}
        ]
        
        for i, match in enumerate(sample_matches, 1):
            try:
                prediction = predictor.predict_match(
                    match["team1_name"], 
                    match["team2_name"],
                    match.get("team1_country"),
                    match.get("team2_country")
                )
                print(f"   Match {i}: {match['team1_name']} vs {match['team2_name']}")
                print(f"     Predicted Winner: {prediction['predicted_winner']}")
                print(f"     Confidence: {prediction['confidence']:.2%}")
            except Exception as e:
                print(f"   Match {i}: Error - {e}")
        
        # Step 7: Show team analysis
        print("\n7. Team analysis...")
        top_teams = sorted(team_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:3]
        for i, (team_name, stats) in enumerate(top_teams, 1):
            print(f"   {i}. {team_name}: {stats['win_rate']:.1%} win rate ({stats['total_matches']} matches)")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY! 🎮")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'python main.py --mode full' for the complete pipeline with real data")
        print("2. Use 'jupyter notebook valorant_analysis.ipynb' for interactive analysis")
        print("3. Check the generated plots and model files")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    run_demo()
