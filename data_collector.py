"""
Data collection module for Valorant Esports API
"""

import requests
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
from config import ENDPOINTS, DATA_CONFIG


class ValorantDataCollector:
    """Class to collect data from Valorant Esports APIs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Valorant-ML-Predictor/1.0'
        })
    
    def fetch_results(self, limit: int = None) -> List[Dict]:
        """Fetch match results from the API"""
        try:
            url = ENDPOINTS['results']
            params = {}
            if limit:
                params['limit'] = limit
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'OK':
                return data.get('data', [])
            else:
                print(f"API returned error status: {data.get('status')}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching results: {e}")
            return []
    
    def fetch_teams(self) -> List[Dict]:
        """Fetch team data from the API"""
        try:
            response = self.session.get(ENDPOINTS['teams'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching teams: {e}")
            return []
    
    def fetch_players(self) -> List[Dict]:
        """Fetch player data from the API"""
        try:
            response = self.session.get(ENDPOINTS['players'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching players: {e}")
            return []
    
    def fetch_events(self, status: str = "all", region: str = "all", page: int = 1) -> List[Dict]:
        """Fetch events data from the API with filtering options"""
        try:
            params = {
                'page': page,
                'status': status,
                'region': region
            }
            response = self.session.get(ENDPOINTS['events'], params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'OK':
                return data.get('data', [])
            else:
                print(f"API returned error status: {data.get('status')}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching events: {e}")
            return []
    
    def fetch_all_events(self, status: str = "all", region: str = "all") -> List[Dict]:
        """Fetch all events data across all pages"""
        all_events = []
        page = 1
        
        while True:
            print(f"Fetching events page {page}...")
            events = self.fetch_events(status, region, page)
            
            if not events:
                break
                
            all_events.extend(events)
            page += 1
            
            # Add delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"Fetched {len(all_events)} events total")
        return all_events
    
    def fetch_matches(self, page: int = 1) -> List[Dict]:
        """Fetch matches data from the API"""
        try:
            params = {'page': page}
            response = self.session.get(ENDPOINTS['matches'], params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'OK':
                return data.get('data', [])
            else:
                print(f"API returned error status: {data.get('status')}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching matches: {e}")
            return []
    
    def fetch_all_matches(self) -> List[Dict]:
        """Fetch all matches data across all pages"""
        all_matches = []
        page = 1
        
        while True:
            print(f"Fetching matches page {page}...")
            matches = self.fetch_matches(page)
            
            if not matches:
                break
                
            all_matches.extend(matches)
            page += 1
            
            # Add delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"Fetched {len(all_matches)} matches total")
        return all_matches
    
    def collect_all_data(self, fetch_all_events: bool = True, fetch_all_matches: bool = True) -> Dict[str, List[Dict]]:
        """Collect all available data from the APIs"""
        print("Collecting data from Valorant Esports APIs...")
        
        data = {}
        
        # Fetch results (most important for our model)
        print("Fetching match results...")
        data['results'] = self.fetch_results(DATA_CONFIG['max_results'])
        print(f"Fetched {len(data['results'])} match results")
        
        # Add small delay to be respectful to the API
        time.sleep(1)
        
        # Fetch other data
        print("Fetching teams data...")
        data['teams'] = self.fetch_teams()
        print(f"Fetched {len(data['teams'])} teams")
        
        time.sleep(1)
        
        print("Fetching players data...")
        data['players'] = self.fetch_players()
        print(f"Fetched {len(data['players'])} players")
        
        time.sleep(1)
        
        # Fetch events data (enhanced with pagination)
        if fetch_all_events:
            print("Fetching all events data...")
            data['events'] = self.fetch_all_events()
        else:
            print("Fetching events data...")
            data['events'] = self.fetch_events()
        print(f"Fetched {len(data['events'])} events")
        
        time.sleep(1)
        
        # Fetch matches data (enhanced with pagination)
        if fetch_all_matches:
            print("Fetching all matches data...")
            data['matches'] = self.fetch_all_matches()
        else:
            print("Fetching matches data...")
            data['matches'] = self.fetch_matches()
        print(f"Fetched {len(data['matches'])} matches")
        
        return data
    
    def save_data(self, data: Dict[str, List[Dict]], filename_prefix: str = "valorant_data"):
        """Save collected data to JSON and CSV files"""
        import json
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data_list in data.items():
            # Save as JSON
            json_filename = f"{filename_prefix}_{data_type}_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)
            print(f"Saved {data_type} data to {json_filename}")
            
            # Save as CSV if data is not empty
            if data_list:
                try:
                    df = pd.DataFrame(data_list)
                    csv_filename = f"{filename_prefix}_{data_type}_{timestamp}.csv"
                    df.to_csv(csv_filename, index=False, encoding='utf-8')
                    print(f"Saved {data_type} data to {csv_filename}")
                except Exception as e:
                    print(f"Warning: Could not save {data_type} as CSV: {e}")
    
    def save_events_to_csv(self, events: List[Dict], filename: str = None):
        """Save events data to CSV with proper formatting"""
        if not events:
            print("No events data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"valorant_events_{timestamp}.csv"
        
        # Create DataFrame and clean data
        df = pd.DataFrame(events)
        
        # Rename columns for better readability
        column_mapping = {
            'id': 'event_id',
            'name': 'event_name',
            'status': 'event_status',
            'prizepool': 'prize_pool',
            'dates': 'event_dates',
            'country': 'event_country',
            'img': 'event_image_url'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert prize pool to numeric
        df['prize_pool'] = pd.to_numeric(df['prize_pool'], errors='coerce')
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Events data saved to {filename}")
        print(f"Total events: {len(df)}")
        print(f"Events by status: {df['event_status'].value_counts().to_dict()}")
        
        return df
    
    def save_matches_to_csv(self, matches: List[Dict], filename: str = None):
        """Save matches data to CSV with proper formatting"""
        if not matches:
            print("No matches data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"valorant_matches_{timestamp}.csv"
        
        # Process matches data for CSV
        processed_matches = []
        
        for match in matches:
            # Extract team information
            teams = match.get('teams', [])
            if len(teams) >= 2:
                team1 = teams[0]
                team2 = teams[1]
                
                processed_match = {
                    'match_id': match.get('id'),
                    'team1_name': team1.get('name'),
                    'team1_country': team1.get('country'),
                    'team1_score': team1.get('score'),
                    'team2_name': team2.get('name'),
                    'team2_country': team2.get('country'),
                    'team2_score': team2.get('score'),
                    'status': match.get('status'),
                    'event': match.get('event'),
                    'tournament': match.get('tournament'),
                    'image_url': match.get('img'),
                    'time_until': match.get('in'),
                    'timestamp': match.get('timestamp'),
                    'utc_date': match.get('utcDate'),
                    'utc': match.get('utc')
                }
                
                processed_matches.append(processed_match)
        
        # Create DataFrame
        df = pd.DataFrame(processed_matches)
        
        # Convert timestamp to datetime if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['match_datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Matches data saved to {filename}")
        print(f"Total matches: {len(df)}")
        print(f"Matches by status: {df['status'].value_counts().to_dict()}")
        
        return df


if __name__ == "__main__":
    collector = ValorantDataCollector()
    data = collector.collect_all_data()
    collector.save_data(data)
