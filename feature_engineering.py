"""
Feature engineering module for Valorant match prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import MODEL_CONFIG


class ValorantFeatureEngineer:
    """Class to engineer features for machine learning model"""
    
    def __init__(self, team_stats: Dict, region_strength: Dict, h2h_stats: Dict):
        self.team_stats = team_stats
        self.region_strength = region_strength
        self.h2h_stats = h2h_stats
        
    def get_team_stat(self, team_name: str, stat_name: str, default_value: float = 0.0) -> float:
        """Get a specific statistic for a team with default fallback"""
        if team_name in self.team_stats:
            return self.team_stats[team_name].get(stat_name, default_value)
        return default_value
    
    def get_region_strength(self, country: str, default_value: float = 0.5) -> float:
        """Get regional strength for a country with default fallback"""
        return self.region_strength.get(country, default_value)
    
    def get_h2h_advantage(self, team1: str, team2: str) -> float:
        """Get head-to-head advantage for team1 over team2"""
        # Try both orderings of the team pair
        key1 = (team1, team2)
        key2 = (team2, team1)
        
        if key1 in self.h2h_stats:
            return self.h2h_stats[key1]['team1_win_rate']
        elif key2 in self.h2h_stats:
            return 1.0 - self.h2h_stats[key2]['team1_win_rate']
        else:
            return 0.5  # No H2H data, assume equal
    
    def create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for each match in the dataset"""
        features = []
        
        for _, match in df.iterrows():
            team1_name = match['team1_name']
            team2_name = match['team2_name']
            team1_country = match['team1_country']
            team2_country = match['team2_country']
            
            # Skip if team names are missing
            if pd.isna(team1_name) or pd.isna(team2_name):
                continue
            
            # Basic team statistics
            team1_win_rate = self.get_team_stat(team1_name, 'win_rate')
            team2_win_rate = self.get_team_stat(team2_name, 'win_rate')
            
            team1_recent_form = self.get_team_stat(team1_name, 'recent_form')
            team2_recent_form = self.get_team_stat(team2_name, 'recent_form')
            
            team1_avg_score = self.get_team_stat(team1_name, 'avg_score')
            team2_avg_score = self.get_team_stat(team2_name, 'avg_score')
            
            # Regional strength
            team1_region_strength = self.get_region_strength(team1_country)
            team2_region_strength = self.get_region_strength(team2_country)
            
            # Head-to-head advantage
            h2h_advantage = self.get_h2h_advantage(team1_name, team2_name)
            
            # Additional derived features
            win_rate_diff = team1_win_rate - team2_win_rate
            form_diff = team1_recent_form - team2_recent_form
            score_diff = team1_avg_score - team2_avg_score
            region_diff = team1_region_strength - team2_region_strength
            
            # Tournament importance (VCT events are more important)
            tournament_importance = 1.0
            tournament = str(match.get('tournament', '')).lower()
            if 'champions' in tournament or 'masters' in tournament:
                tournament_importance = 3.0
            elif 'challengers' in tournament:
                tournament_importance = 2.0
            elif 'game changers' in tournament:
                tournament_importance = 1.5
            
            # Match recency (more recent matches are more relevant)
            days_ago = (pd.Timestamp.now() - match['match_date']).days
            recency_factor = max(0.1, 1.0 - (days_ago / 365.0))  # Decay over a year
            
            feature_row = {
                'match_id': match['match_id'],
                'team1_name': team1_name,
                'team2_name': team2_name,
                'team1_win_rate': team1_win_rate,
                'team2_win_rate': team2_win_rate,
                'team1_recent_form': team1_recent_form,
                'team2_recent_form': team2_recent_form,
                'team1_avg_score': team1_avg_score,
                'team2_avg_score': team2_avg_score,
                'team1_region_strength': team1_region_strength,
                'team2_region_strength': team2_region_strength,
                'head_to_head_advantage': h2h_advantage,
                'win_rate_diff': win_rate_diff,
                'form_diff': form_diff,
                'score_diff': score_diff,
                'region_diff': region_diff,
                'tournament_importance': tournament_importance,
                'recency_factor': recency_factor,
                'team1_won': match['team1_won'],
                'team2_won': match['team2_won'],
                'match_date': match['match_date'],
                'tournament': match['tournament'],
                'event': match['event']
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction accuracy"""
        advanced_features = df.copy()
        
        # Momentum features (performance in last 5 matches)
        for team_col in ['team1_name', 'team2_name']:
            momentum_col = f"{team_col.split('_')[0]}_momentum"
            advanced_features[momentum_col] = 0.0
            
            for idx, row in advanced_features.iterrows():
                team_name = row[team_col]
                match_date = row['match_date']
                
                # Get recent matches for this team
                recent_matches = df[
                    ((df['team1_name'] == team_name) | (df['team2_name'] == team_name)) &
                    (df['match_date'] < match_date)
                ].nlargest(5, 'match_date')
                
                if len(recent_matches) > 0:
                    wins = 0
                    for _, recent_match in recent_matches.iterrows():
                        is_team1 = recent_match['team1_name'] == team_name
                        if (is_team1 and recent_match['team1_won']) or (not is_team1 and recent_match['team2_won']):
                            wins += 1
                    
                    momentum = wins / len(recent_matches)
                    advanced_features.loc[idx, momentum_col] = momentum
        
        # Strength of schedule (average opponent strength)
        for team_col in ['team1_name', 'team2_name']:
            sos_col = f"{team_col.split('_')[0]}_strength_of_schedule"
            advanced_features[sos_col] = 0.0
            
            for idx, row in advanced_features.iterrows():
                team_name = row[team_col]
                match_date = row['match_date']
                
                # Get recent opponents
                recent_matches = df[
                    ((df['team1_name'] == team_name) | (df['team2_name'] == team_name)) &
                    (df['match_date'] < match_date)
                ].nlargest(10, 'match_date')
                
                if len(recent_matches) > 0:
                    opponent_strengths = []
                    for _, recent_match in recent_matches.iterrows():
                        is_team1 = recent_match['team1_name'] == team_name
                        opponent = recent_match['team2_name'] if is_team1 else recent_match['team1_name']
                        opponent_strength = self.get_team_stat(opponent, 'win_rate')
                        opponent_strengths.append(opponent_strength)
                    
                    sos = np.mean(opponent_strengths) if opponent_strengths else 0.5
                    advanced_features.loc[idx, sos_col] = sos
        
        # Tournament experience (number of major tournaments played)
        for team_col in ['team1_name', 'team2_name']:
            exp_col = f"{team_col.split('_')[0]}_tournament_experience"
            advanced_features[exp_col] = 0.0
            
            for idx, row in advanced_features.iterrows():
                team_name = row[team_col]
                match_date = row['match_date']
                
                # Count major tournaments before this match
                major_tournaments = df[
                    ((df['team1_name'] == team_name) | (df['team2_name'] == team_name)) &
                    (df['match_date'] < match_date) &
                    (df['tournament'].str.contains('Champions|Masters', case=False, na=False))
                ]['tournament'].nunique()
                
                advanced_features.loc[idx, exp_col] = major_tournaments
        
        return advanced_features
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare final features for machine learning model"""
        # Create basic features
        features_df = self.create_match_features(df)
        
        # Create advanced features
        features_df = self.create_advanced_features(features_df)
        
        # Select features for ML model
        feature_columns = MODEL_CONFIG['feature_columns'] + [
            'win_rate_diff', 'form_diff', 'score_diff', 'region_diff',
            'tournament_importance', 'recency_factor',
            'team1_momentum', 'team2_momentum',
            'team1_strength_of_schedule', 'team2_strength_of_schedule',
            'team1_tournament_experience', 'team2_tournament_experience'
        ]
        
        # Filter to only include available columns
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        # Create X (features) and y (target)
        X = features_df[available_features].copy()
        y = features_df['team1_won'].astype(int)
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Add match metadata for reference
        metadata_columns = ['match_id', 'team1_name', 'team2_name', 'match_date', 'tournament', 'event']
        metadata = features_df[metadata_columns].copy()
        
        return X, y, metadata, available_features
    
    def create_prediction_features(self, team1_name: str, team2_name: str, 
                                 team1_country: str = None, team2_country: str = None) -> pd.Series:
        """Create features for predicting a specific match"""
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
        
        # Create feature series
        features = pd.Series({
            'team1_win_rate': team1_win_rate,
            'team2_win_rate': team2_win_rate,
            'team1_recent_form': team1_recent_form,
            'team2_recent_form': team2_recent_form,
            'team1_avg_score': team1_avg_score,
            'team2_avg_score': team2_avg_score,
            'team1_region_strength': team1_region_strength,
            'team2_region_strength': team2_region_strength,
            'head_to_head_advantage': h2h_advantage,
            'win_rate_diff': win_rate_diff,
            'form_diff': form_diff,
            'score_diff': score_diff,
            'region_diff': region_diff,
            'tournament_importance': 3.0,  # Assume VCT importance
            'recency_factor': 1.0  # Current prediction
        })
        
        return features


if __name__ == "__main__":
    from data_collector import ValorantDataCollector
    from data_preprocessor import ValorantDataPreprocessor
    
    # Collect and preprocess data
    collector = ValorantDataCollector()
    raw_data = collector.collect_all_data()
    
    preprocessor = ValorantDataPreprocessor()
    df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
    
    # Engineer features
    feature_engineer = ValorantFeatureEngineer(team_stats, region_strength, h2h_stats)
    X, y, metadata, feature_names = feature_engineer.prepare_ml_features(df)
    
    print(f"Created {len(X)} samples with {len(feature_names)} features")
    print(f"Feature names: {feature_names}")
    print(f"Target distribution: {y.value_counts()}")
