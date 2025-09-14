"""
Data preprocessing module for Valorant match data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
from config import DATA_CONFIG


class ValorantDataPreprocessor:
    """Class to preprocess and clean Valorant match data"""
    
    def __init__(self):
        self.team_stats = {}
        self.region_strength = {}
        
    def parse_time_ago(self, time_str: str) -> datetime:
        """Parse time ago string to datetime object"""
        now = datetime.now()
        
        if 'h' in time_str and 'm' in time_str:
            # Format: "2h 15m"
            hours = int(re.search(r'(\d+)h', time_str).group(1))
            minutes = int(re.search(r'(\d+)m', time_str).group(1))
            return now - timedelta(hours=hours, minutes=minutes)
        elif 'h' in time_str:
            # Format: "5h"
            hours = int(re.search(r'(\d+)h', time_str).group(1))
            return now - timedelta(hours=hours)
        elif 'd' in time_str:
            # Format: "1d 2h" or "2d"
            days = int(re.search(r'(\d+)d', time_str).group(1))
            if 'h' in time_str:
                hours = int(re.search(r'(\d+)h', time_str).group(1))
                return now - timedelta(days=days, hours=hours)
            else:
                return now - timedelta(days=days)
        elif 'w' in time_str:
            # Format: "1w 1d" or "1w"
            weeks = int(re.search(r'(\d+)w', time_str).group(1))
            if 'd' in time_str:
                days = int(re.search(r'(\d+)d', time_str).group(1))
                return now - timedelta(weeks=weeks, days=days)
            else:
                return now - timedelta(weeks=weeks)
        else:
            return now
    
    def clean_match_data(self, results: List[Dict]) -> pd.DataFrame:
        """Clean and structure match results data"""
        matches = []
        
        for result in results:
            try:
                # Parse teams data
                teams = result.get('teams', [])
                if len(teams) != 2:
                    continue
                
                team1 = teams[0]
                team2 = teams[1]
                
                # Parse scores
                team1_score = int(team1.get('score', 0))
                team2_score = int(team2.get('score', 0))
                
                # Determine winner
                team1_won = team1.get('won', False)
                team2_won = team2.get('won', False)
                
                # Parse match date
                match_date = self.parse_time_ago(result.get('ago', '0h'))
                
                match_data = {
                    'match_id': result.get('id'),
                    'team1_name': team1.get('name'),
                    'team2_name': team2.get('name'),
                    'team1_country': team1.get('country'),
                    'team2_country': team2.get('country'),
                    'team1_score': team1_score,
                    'team2_score': team2_score,
                    'team1_won': team1_won,
                    'team2_won': team2_won,
                    'winner': team1.get('name') if team1_won else team2.get('name'),
                    'match_date': match_date,
                    'status': result.get('status'),
                    'event': result.get('event'),
                    'tournament': result.get('tournament')
                }
                
                matches.append(match_data)
                
            except (ValueError, KeyError, AttributeError) as e:
                print(f"Error processing match {result.get('id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(matches)
    
    def calculate_team_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate comprehensive team statistics"""
        team_stats = {}
        
        for team in df['team1_name'].unique():
            if pd.isna(team):
                continue
                
            # Get all matches for this team
            team_matches = df[
                (df['team1_name'] == team) | (df['team2_name'] == team)
            ].copy()
            
            if len(team_matches) < DATA_CONFIG['min_matches_for_stats']:
                continue
            
            # Calculate wins and losses
            wins = 0
            total_matches = len(team_matches)
            total_score = 0
            total_opponent_score = 0
            
            recent_matches = team_matches.nlargest(DATA_CONFIG['recent_matches_window'], 'match_date')
            recent_wins = 0
            
            for _, match in team_matches.iterrows():
                is_team1 = match['team1_name'] == team
                team_score = match['team1_score'] if is_team1 else match['team2_score']
                opponent_score = match['team2_score'] if is_team1 else match['team1_score']
                team_won = match['team1_won'] if is_team1 else match['team2_won']
                
                total_score += team_score
                total_opponent_score += opponent_score
                
                if team_won:
                    wins += 1
                
                # Check if this is a recent match
                if match['match_id'] in recent_matches['match_id'].values:
                    if team_won:
                        recent_wins += 1
            
            # Calculate statistics
            win_rate = wins / total_matches if total_matches > 0 else 0
            avg_score = total_score / total_matches if total_matches > 0 else 0
            avg_opponent_score = total_opponent_score / total_matches if total_matches > 0 else 0
            recent_form = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0
            
            team_stats[team] = {
                'total_matches': total_matches,
                'wins': wins,
                'losses': total_matches - wins,
                'win_rate': win_rate,
                'avg_score': avg_score,
                'avg_opponent_score': avg_opponent_score,
                'score_difference': avg_score - avg_opponent_score,
                'recent_form': recent_form,
                'recent_matches': len(recent_matches)
            }
        
        return team_stats
    
    def calculate_region_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate regional strength based on team performance"""
        region_stats = {}
        
        for country in df['team1_country'].unique():
            if pd.isna(country):
                continue
                
            # Get all teams from this region
            region_teams = df[
                (df['team1_country'] == country) | (df['team2_country'] == country)
            ]['team1_name'].unique()
            
            region_wins = 0
            region_matches = 0
            
            for team in region_teams:
                if pd.isna(team):
                    continue
                    
                team_matches = df[
                    (df['team1_name'] == team) | (df['team2_name'] == team)
                ]
                
                for _, match in team_matches.iterrows():
                    is_team1 = match['team1_name'] == team
                    team_won = match['team1_won'] if is_team1 else match['team2_won']
                    
                    region_matches += 1
                    if team_won:
                        region_wins += 1
            
            region_win_rate = region_wins / region_matches if region_matches > 0 else 0
            region_stats[country] = region_win_rate
        
        return region_stats
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> Dict[Tuple[str, str], Dict]:
        """Calculate head-to-head statistics between teams"""
        h2h_stats = {}
        
        # Get all unique team pairs
        team_pairs = set()
        for _, match in df.iterrows():
            team1 = match['team1_name']
            team2 = match['team2_name']
            if pd.notna(team1) and pd.notna(team2):
                pair = tuple(sorted([team1, team2]))
                team_pairs.add(pair)
        
        for team1, team2 in team_pairs:
            # Get all matches between these teams
            matches = df[
                ((df['team1_name'] == team1) & (df['team2_name'] == team2)) |
                ((df['team1_name'] == team2) & (df['team2_name'] == team1))
            ]
            
            if len(matches) == 0:
                continue
            
            team1_wins = 0
            team2_wins = 0
            
            for _, match in matches.iterrows():
                if match['team1_name'] == team1 and match['team1_won']:
                    team1_wins += 1
                elif match['team2_name'] == team1 and match['team2_won']:
                    team1_wins += 1
                else:
                    team2_wins += 1
            
            total_matches = len(matches)
            h2h_stats[(team1, team2)] = {
                'team1_wins': team1_wins,
                'team2_wins': team2_wins,
                'total_matches': total_matches,
                'team1_win_rate': team1_wins / total_matches if total_matches > 0 else 0.5
            }
        
        return h2h_stats
    
    def preprocess_data(self, raw_data: Dict[str, List[Dict]]) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Clean match data
        print("Cleaning match data...")
        df = self.clean_match_data(raw_data['results'])
        print(f"Processed {len(df)} matches")
        
        # Calculate team statistics
        print("Calculating team statistics...")
        self.team_stats = self.calculate_team_statistics(df)
        print(f"Calculated stats for {len(self.team_stats)} teams")
        
        # Calculate region strength
        print("Calculating regional strength...")
        self.region_strength = self.calculate_region_strength(df)
        print(f"Calculated strength for {len(self.region_strength)} regions")
        
        # Calculate head-to-head statistics
        print("Calculating head-to-head statistics...")
        h2h_stats = self.calculate_head_to_head(df)
        print(f"Calculated H2H for {len(h2h_stats)} team pairs")
        
        return df, self.team_stats, self.region_strength, h2h_stats
    
    def save_processed_data(self, df: pd.DataFrame, team_stats: Dict, region_strength: Dict, 
                          h2h_stats: Dict, filename_prefix: str = "processed_valorant_data"):
        """Save processed data to files"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame
        df.to_csv(f"{filename_prefix}_matches_{timestamp}.csv", index=False)
        print(f"Saved processed matches to {filename_prefix}_matches_{timestamp}.csv")
        
        # Save team statistics
        with open(f"{filename_prefix}_team_stats_{timestamp}.json", 'w') as f:
            json.dump(team_stats, f, indent=2, default=str)
        print(f"Saved team statistics to {filename_prefix}_team_stats_{timestamp}.json")
        
        # Save region strength
        with open(f"{filename_prefix}_region_strength_{timestamp}.json", 'w') as f:
            json.dump(region_strength, f, indent=2)
        print(f"Saved region strength to {filename_prefix}_region_strength_{timestamp}.json")
        
        # Save H2H statistics
        with open(f"{filename_prefix}_h2h_stats_{timestamp}.json", 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            h2h_serializable = {f"{k[0]}_vs_{k[1]}": v for k, v in h2h_stats.items()}
            json.dump(h2h_serializable, f, indent=2)
        print(f"Saved H2H statistics to {filename_prefix}_h2h_stats_{timestamp}.json")


if __name__ == "__main__":
    from data_collector import ValorantDataCollector
    
    # Collect and preprocess data
    collector = ValorantDataCollector()
    raw_data = collector.collect_all_data()
    
    preprocessor = ValorantDataPreprocessor()
    df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
    
    # Save processed data
    preprocessor.save_processed_data(df, team_stats, region_strength, h2h_stats)
