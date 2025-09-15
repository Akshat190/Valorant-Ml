"""
Data collection module for Valorant Esports API
"""

import requests
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
from config import ENDPOINTS, DATA_CONFIG, VLRGG_ENDPOINTS


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

    def fetch_results_full(self, region: str = 'all', timespan: str = 'all', q: str = 'all', max_pages: int = 20) -> List[Dict]:
        """Fetch results with filters and pagination from unofficial VLR API"""
        all_rows: List[Dict] = []
        page = 1
        while page <= max_pages:
            try:
                params = {
                    'region': region,
                    'timespan': timespan,
                    'q': q,
                    'page': page
                }
                resp = self.session.get(ENDPOINTS['results'], params=params)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get('data', []) if isinstance(data, dict) else []
                if not rows:
                    break
                # tag page for traceability
                for r in rows:
                    if isinstance(r, dict):
                        r['page'] = page
                        r['region'] = region
                        r['timespan'] = timespan
                        r['q'] = q
                all_rows.extend(rows)
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"Error fetching results page {page}: {e}")
                break
        print(f"Fetched {len(all_rows)} results (region={region}, timespan={timespan}, q={q})")
        return all_rows
    
    def fetch_teams(self) -> List[Dict]:
        """Fetch team data from the API - DISABLED due to 404 errors"""
        print("Teams API disabled - returning empty list")
        return []
    
    def fetch_players(self) -> List[Dict]:
        """Fetch player data from the API - DISABLED due to 404 errors"""
        print("Players API disabled - returning empty list")
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
    
    def fetch_all_events(self, status: str = "all", region: str = "all", max_pages: int = None) -> List[Dict]:
        """Fetch events data across pages with limit"""
        if max_pages is None:
            max_pages = DATA_CONFIG['max_events_pages']
            
        all_events = []
        page = 1
        
        while page <= max_pages:
            print(f"Fetching events page {page}...")
            events = self.fetch_events(status, region, page)
            
            if not events:
                print(f"No more events found on page {page}")
                break
                
            all_events.extend(events)
            page += 1
            
            # Add delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"Fetched {len(all_events)} events total (max {max_pages} pages)")
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

    def fetch_all_matches_bulk(self) -> List[Dict]:
        """Fetch all matches from the bulk endpoint"""
        try:
            response = self.session.get(ENDPOINTS['matches_all'])
            response.raise_for_status()
            data = response.json()
            items = data.get('data') if isinstance(data, dict) else []
            print(f"Fetched {len(items)} from matches/get-all-matches")
            return items if isinstance(items, list) else []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching get-all-matches: {e}")
            return []
    
    def fetch_all_matches(self, max_pages: int = None) -> List[Dict]:
        """Fetch matches data across pages with limit"""
        if max_pages is None:
            max_pages = DATA_CONFIG['max_matches_pages']
            
        all_matches = []
        page = 1
        
        while page <= max_pages:
            print(f"Fetching matches page {page}...")
            matches = self.fetch_matches(page)
            
            if not matches:
                print(f"No more matches found on page {page}")
                break
                
            all_matches.extend(matches)
            page += 1
            
            # Add delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"Fetched {len(all_matches)} matches total (max {max_pages} pages)")
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
        
        # Skip teams and players (APIs returning 404)
        print("Skipping teams and players (APIs not available)...")
        data['teams'] = []
        data['players'] = []
        
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
            # Combine paged and bulk for best coverage
            paged = self.fetch_all_matches()
            bulk = self.fetch_all_matches_bulk()
            data['matches'] = (paged or []) + (bulk or [])
        else:
            print("Fetching matches data...")
            data['matches'] = self.fetch_matches()
        print(f"Fetched {len(data['matches'])} matches")
        
        return data

    def collect_unofficial_full(self) -> Dict[str, List[Dict]]:
        """Collect comprehensive datasets using the user-specified unofficial endpoints"""
        print("Collecting data using unofficial endpoints (results/events/matches bulk)...")
        collected: Dict[str, List[Dict]] = {}

        # Results across regions and pages
        regions = ['all', 'na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc']
        timespans = ['all', '30']
        q_values = ['all', 'completed', 'upcoming']
        all_results: List[Dict] = []
        for region in regions:
            for timespan in timespans:
                for q in q_values:
                    rows = self.fetch_results_full(region=region, timespan=timespan, q=q, max_pages=50)
                    all_results.extend(rows)
        collected['results'] = all_results

        # Events across statuses and pages
        all_events: List[Dict] = []
        for status in ['all', 'ongoing', 'upcoming', 'completed']:
            page = 1
            while page <= DATA_CONFIG['max_events_pages']:
                try:
                    params = {'status': status, 'region': 'all', 'page': page}
                    resp = self.session.get(ENDPOINTS['events'], params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                    items = payload.get('data', []) if isinstance(payload, dict) else []
                    if not items:
                        break
                    for it in items:
                        if isinstance(it, dict):
                            it['status_filter'] = status
                            it['page'] = page
                    all_events.extend(items)
                    page += 1
                    time.sleep(0.3)
                except Exception as e:
                    print(f"Error fetching events status={status} page={page}: {e}")
                    break
        collected['events'] = all_events

        # Bulk matches
        collected['matches'] = self.fetch_all_matches_bulk()

        return collected

    # ---------------- VLRGG Extended Collection ----------------
    def vlrgg_get(self, key: str, params: Optional[Dict] = None) -> Dict:
        """Generic GET to VLRGG endpoints with fallback URL patterns"""
        params = params or {}
        url_candidates = []
        try:
            # Configured endpoint first
            url_candidates.append(VLRGG_ENDPOINTS[key])
        except KeyError:
            pass

        # Fallback patterns based on public docs variants
        base_variants = [
            "https://vlrggapi.vercel.app/api",
            "https://vlrggapi.vercel.app/api/v1",
            "https://vlrggapi.vercel.app",
            "https://vlrggapi.vercel.app/v1",
        ]
        path_variants = [key, key.strip('/')]

        for base in base_variants:
            for path in path_variants:
                url_candidates.append(f"{base}/{path}")

        last_err = None
        for url in url_candidates:
            try:
                resp = self.session.get(url, params=params)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                continue

        if last_err:
            print(f"VLRGG {key} fetch error: {last_err}")
        else:
            print(f"VLRGG {key} fetch error: No valid URL patterns responded")
        return {}

    def collect_vlrgg_all(self) -> Dict[str, List[Dict]]:
        """Collect comprehensive datasets from VLRGG API with parameters and pagination"""
        print("Collecting from VLRGG API (news, stats, rankings, events, match)...")
        data: Dict[str, List[Dict]] = {}

        # News (no params)
        news_payload = self.vlrgg_get('news')
        news_items = news_payload.get('data') if isinstance(news_payload, dict) else []
        data['news'] = news_items if isinstance(news_items, list) else []
        print(f"Fetched {len(data['news'])} from news")
        time.sleep(0.2)

        # Regions and timespans
        regions = ['na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc']
        timespans = ['30', 'all']

        # Stats (region, timespan)
        stats_all: List[Dict] = []
        for region in regions:
            for timespan in timespans:
                payload = self.vlrgg_get('stats', {'region': region, 'timespan': timespan})
                items = payload.get('data') if isinstance(payload, dict) else []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            it['region'] = region
                            it['timespan'] = timespan
                    stats_all.extend(items)
                time.sleep(0.15)
        data['stats'] = stats_all
        print(f"Fetched {len(stats_all)} rows from stats (all regions/timespans)")

        # Rankings (region)
        rankings_all: List[Dict] = []
        for region in regions:
            payload = self.vlrgg_get('rankings', {'region': region})
            items = payload.get('data') if isinstance(payload, dict) else []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it['region'] = region
                rankings_all.extend(items)
            time.sleep(0.15)
        data['rankings'] = rankings_all
        print(f"Fetched {len(rankings_all)} rows from rankings (all regions)")

        # Matches (q: upcoming, live_score, results)
        match_all: List[Dict] = []
        for q in ['upcoming', 'live_score', 'results']:
            payload = self.vlrgg_get('match', {'q': q})
            items = payload.get('data') if isinstance(payload, dict) else []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it['q'] = q
                match_all.extend(items)
            time.sleep(0.15)
        data['match'] = match_all
        print(f"Fetched {len(match_all)} rows from match (all q types)")

        # Events (q filter + pagination for completed)
        events_all: List[Dict] = []
        # Upcoming and both (no page)
        for q in [None, 'upcoming']:
            params = {'q': q} if q else None
            payload = self.vlrgg_get('events', params)
            items = payload.get('data') if isinstance(payload, dict) else []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it['q'] = q or 'all'
                        it['page'] = 1
                events_all.extend(items)
            time.sleep(0.15)
        # Completed with pages
        for page in range(1, 11):  # first 10 pages
            payload = self.vlrgg_get('events', {'q': 'completed', 'page': page})
            items = payload.get('data') if isinstance(payload, dict) else []
            if not items:
                # Stop if no more items
                break
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it['q'] = 'completed'
                        it['page'] = page
                events_all.extend(items)
            time.sleep(0.15)
        data['events'] = events_all
        print(f"Fetched {len(events_all)} rows from events (all filters)")

        # Health
        health_payload = self.vlrgg_get('health')
        health_items = health_payload.get('data') if isinstance(health_payload, dict) else []
        data['health'] = health_items if isinstance(health_items, list) else []
        print(f"Fetched {len(data['health'])} from health")

        return data

    def save_vlrgg_csv(self, vlrgg_data: Dict[str, List[Dict]]):
        """Save VLRGG datasets to CSVs under data/raw/ with schema-friendly columns"""
        import os
        from datetime import datetime
        os.makedirs("data/raw", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, records in vlrgg_data.items():
            try:
                # Ensure list of dicts
                recs = records if isinstance(records, list) else ([records] if records else [])
                df = pd.json_normalize(recs)
                out = f"data/raw/vlrgg_{name}_{ts}.csv"
                df.to_csv(out, index=False, encoding='utf-8')
                print(f"Saved {name} -> {out} ({len(df)})")
            except Exception as e:
                print(f"Could not save {name}: {e}")
    
    def save_data(self, data: Dict[str, List[Dict]], filename_prefix: str = "valorant_data"):
        """Save collected data to JSON and CSV files in organized folders"""
        import json
        import pandas as pd
        from datetime import datetime
        import os
        
        # Create data/raw directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data_list in data.items():
            # Save as JSON in data/raw folder
            json_filename = f"data/raw/{filename_prefix}_{data_type}_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)
            print(f"Saved {data_type} data to {json_filename}")
            
            # Save as CSV if data is not empty
            if data_list:
                try:
                    df = pd.DataFrame(data_list)
                    csv_filename = f"data/raw/{filename_prefix}_{data_type}_{timestamp}.csv"
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
            filename = f"data/raw/valorant_events_{timestamp}.csv"
        elif not filename.startswith("data/"):
            filename = f"data/raw/{filename}"
        
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
            filename = f"data/raw/valorant_matches_{timestamp}.csv"
        elif not filename.startswith("data/"):
            filename = f"data/raw/{filename}"
        
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
