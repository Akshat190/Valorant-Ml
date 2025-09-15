"""
Prediction interface for Valorant match outcomes
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class ValorantPredictionInterface:
    """Interface for making Valorant match predictions"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.team_stats = {}
        self.region_strength = {}
        self.h2h_stats = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(f"{model_path}.joblib")
            self.scaler = joblib.load(f"{model_path}_scaler.joblib")
            
            # Try to load features from results folder
            features_path = f"{model_path}_features.txt"
            if not os.path.exists(features_path):
                features_path = f"data/results/{os.path.basename(model_path)}_features.txt"
            
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_statistics(self, team_stats_path: str, region_strength_path: str, h2h_stats_path: str):
        """Load team statistics and regional strength data"""
        try:
            with open(team_stats_path, 'r') as f:
                self.team_stats = json.load(f)
            
            with open(region_strength_path, 'r') as f:
                self.region_strength = json.load(f)
            
            with open(h2h_stats_path, 'r') as f:
                h2h_data = json.load(f)
                # Convert string keys back to tuples
                self.h2h_stats = {}
                for key, value in h2h_data.items():
                    team1, team2 = key.split('_vs_')
                    self.h2h_stats[(team1, team2)] = value
            
            print("Statistics loaded successfully")
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def get_team_stat(self, team_name: str, stat_name: str, default_value: float = 0.0) -> float:
        """Get team statistic with fallback"""
        if team_name in self.team_stats:
            return float(self.team_stats[team_name].get(stat_name, default_value))
        return default_value
    
    def get_region_strength(self, country: str, default_value: float = 0.5) -> float:
        """Get regional strength with fallback"""
        return float(self.region_strength.get(country, default_value))
    
    def get_h2h_advantage(self, team1: str, team2: str) -> float:
        """Get head-to-head advantage"""
        key1 = (team1, team2)
        key2 = (team2, team1)
        
        if key1 in self.h2h_stats:
            return float(self.h2h_stats[key1]['team1_win_rate'])
        elif key2 in self.h2h_stats:
            return 1.0 - float(self.h2h_stats[key2]['team1_win_rate'])
        else:
            return 0.5
    
    def create_prediction_features(self, team1_name: str, team2_name: str,
                                 team1_country: str = None, team2_country: str = None) -> np.ndarray:
        """Create features for prediction"""
        # Basic team statistics
        team1_win_rate = self.get_team_stat(team1_name, 'win_rate')
        team2_win_rate = self.get_team_stat(team2_name, 'win_rate')
        
        team1_recent_form = self.get_team_stat(team1_name, 'recent_form')
        team2_recent_form = self.get_team_stat(team2_name, 'recent_form')
        
        team1_avg_score = self.get_team_stat(team1_name, 'avg_score')
        team2_avg_score = self.get_team_stat(team2_name, 'avg_score')
        
        # Regional strength
        team1_region_strength = self.get_region_strength(team1_country) if team1_country else 0.5
        team2_region_strength = self.get_region_strength(team2_country) if team2_country else 0.5
        
        # Head-to-head advantage
        h2h_advantage = self.get_h2h_advantage(team1_name, team2_name)
        
        # Derived features
        win_rate_diff = team1_win_rate - team2_win_rate
        form_diff = team1_recent_form - team2_recent_form
        score_diff = team1_avg_score - team2_avg_score
        region_diff = team1_region_strength - team2_region_strength
        
        # Additional features to match model expectations
        team1_momentum = self.get_team_stat(team1_name, 'momentum', default_value=0.5)
        team2_momentum = self.get_team_stat(team2_name, 'momentum', default_value=0.5)
        team1_strength_of_schedule = self.get_team_stat(team1_name, 'strength_of_schedule', default_value=0.5)
        team2_strength_of_schedule = self.get_team_stat(team2_name, 'strength_of_schedule', default_value=0.5)
        team1_tournament_experience = self.get_team_stat(team1_name, 'tournament_experience', default_value=0.5)
        team2_tournament_experience = self.get_team_stat(team2_name, 'tournament_experience', default_value=0.5)
        
        # Create feature vector (21 features to match model)
        features = np.array([
            team1_win_rate,
            team2_win_rate,
            team1_recent_form,
            team2_recent_form,
            team1_avg_score,
            team2_avg_score,
            team1_region_strength,
            team2_region_strength,
            h2h_advantage,
            win_rate_diff,
            form_diff,
            score_diff,
            region_diff,
            3.0,  # Tournament importance (VCT)
            1.0,  # Recency factor (current)
            team1_momentum,
            team2_momentum,
            team1_strength_of_schedule,
            team2_strength_of_schedule,
            team1_tournament_experience,
            team2_tournament_experience
        ])
        
        return features
    
    def predict_match(self, team1_name: str, team2_name: str,
                     team1_country: str = None, team2_country: str = None) -> Dict:
        """Predict match outcome"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Create features
        features = self.create_prediction_features(team1_name, team2_name, team1_country, team2_country)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        team1_win_prob = prediction_proba[1]  # Probability of team1 winning
        team2_win_prob = prediction_proba[0]  # Probability of team2 winning
        
        # Determine winner and confidence
        if team1_win_prob > team2_win_prob:
            predicted_winner = team1_name
            confidence = team1_win_prob
        else:
            predicted_winner = team2_name
            confidence = team2_win_prob
        
        return {
            'team1_name': team1_name,
            'team2_name': team2_name,
            'team1_win_probability': float(team1_win_prob),
            'team2_win_probability': float(team2_win_prob),
            'predicted_winner': predicted_winner,
            'confidence': float(confidence),
            'prediction_time': datetime.now().isoformat()
        }
    
    def predict_multiple_matches(self, matches: List[Dict]) -> List[Dict]:
        """Predict multiple matches at once"""
        predictions = []
        
        for match in matches:
            try:
                prediction = self.predict_match(
                    match['team1_name'],
                    match['team2_name'],
                    match.get('team1_country'),
                    match.get('team2_country')
                )
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting match {match}: {e}")
                predictions.append({
                    'error': str(e),
                    'match': match
                })
        
        return predictions
    
    def get_team_analysis(self, team_name: str) -> Dict:
        """Get detailed analysis for a team"""
        if team_name not in self.team_stats:
            return {'error': f'No data available for {team_name}'}
        
        stats = self.team_stats[team_name]
        
        return {
            'team_name': team_name,
            'total_matches': stats.get('total_matches', 0),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'win_rate': stats.get('win_rate', 0.0),
            'recent_form': stats.get('recent_form', 0.0),
            'avg_score': stats.get('avg_score', 0.0),
            'avg_opponent_score': stats.get('avg_opponent_score', 0.0),
            'score_difference': stats.get('score_difference', 0.0)
        }
    
    def compare_teams(self, team1_name: str, team2_name: str) -> Dict:
        """Compare two teams head-to-head"""
        team1_analysis = self.get_team_analysis(team1_name)
        team2_analysis = self.get_team_analysis(team2_name)
        
        if 'error' in team1_analysis or 'error' in team2_analysis:
            return {'error': 'One or both teams not found in database'}
        
        # Calculate advantages
        win_rate_advantage = team1_analysis['win_rate'] - team2_analysis['win_rate']
        form_advantage = team1_analysis['recent_form'] - team2_analysis['recent_form']
        score_advantage = team1_analysis['avg_score'] - team2_analysis['avg_score']
        
        return {
            'team1': team1_analysis,
            'team2': team2_analysis,
            'team1_advantages': {
                'win_rate': win_rate_advantage,
                'recent_form': form_advantage,
                'avg_score': score_advantage
            },
            'team2_advantages': {
                'win_rate': -win_rate_advantage,
                'recent_form': -form_advantage,
                'avg_score': -score_advantage
            }
        }
    
    def get_prediction_explanation(self, team1_name: str, team2_name: str) -> Dict:
        """Get detailed explanation of prediction factors"""
        team1_stats = self.get_team_analysis(team1_name)
        team2_stats = self.get_team_analysis(team2_name)
        
        if 'error' in team1_stats or 'error' in team2_stats:
            return {'error': 'One or both teams not found in database'}
        
        # Calculate key factors
        win_rate_diff = team1_stats['win_rate'] - team2_stats['win_rate']
        form_diff = team1_stats['recent_form'] - team2_stats['recent_form']
        score_diff = team1_stats['avg_score'] - team2_stats['avg_score']
        
        # Determine key advantages
        factors = []
        if abs(win_rate_diff) > 0.1:
            factors.append({
                'factor': 'Overall Win Rate',
                'team1_value': team1_stats['win_rate'],
                'team2_value': team2_stats['win_rate'],
                'advantage': team1_name if win_rate_diff > 0 else team2_name,
                'magnitude': abs(win_rate_diff)
            })
        
        if abs(form_diff) > 0.1:
            factors.append({
                'factor': 'Recent Form',
                'team1_value': team1_stats['recent_form'],
                'team2_value': team2_stats['recent_form'],
                'advantage': team1_name if form_diff > 0 else team2_name,
                'magnitude': abs(form_diff)
            })
        
        if abs(score_diff) > 1.0:
            factors.append({
                'factor': 'Average Score',
                'team1_value': team1_stats['avg_score'],
                'team2_value': team2_stats['avg_score'],
                'advantage': team1_name if score_diff > 0 else team2_name,
                'magnitude': abs(score_diff)
            })
        
        return {
            'team1_name': team1_name,
            'team2_name': team2_name,
            'key_factors': factors,
            'team1_stats': team1_stats,
            'team2_stats': team2_stats
        }


def main():
    """Example usage of the prediction interface"""
    # Initialize prediction interface
    predictor = ValorantPredictionInterface()
    
    # Load model (uncomment when model is trained)
    # predictor.load_model("valorant_model_random_forest")
    # predictor.load_statistics("processed_valorant_data_team_stats.json", 
    #                          "processed_valorant_data_region_strength.json",
    #                          "processed_valorant_data_h2h_stats.json")
    
    # Example predictions (uncomment when model is loaded)
    # prediction = predictor.predict_match("G2 Esports", "Team Heretics", "us", "eu")
    # print("Prediction:", prediction)
    
    print("Valorant Prediction Interface ready!")
    print("Load a trained model to start making predictions.")


if __name__ == "__main__":
    main()
