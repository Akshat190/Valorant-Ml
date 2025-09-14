"""
Specialized script for collecting and storing Valorant events and matches data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from data_collector import ValorantDataCollector


def collect_events_data(status="all", region="all", save_csv=True):
    """Collect events data with filtering options"""
    print("=" * 60)
    print("COLLECTING VALORANT EVENTS DATA")
    print("=" * 60)
    print(f"Status filter: {status}")
    print(f"Region filter: {region}")
    print("=" * 60)
    
    collector = ValorantDataCollector()
    
    # Fetch all events
    events = collector.fetch_all_events(status=status, region=region)
    
    if not events:
        print("No events data found")
        return None
    
    # Save to CSV if requested
    if save_csv:
        events_df = collector.save_events_to_csv(events)
        
        # Display summary
        print("\nEVENTS DATA SUMMARY:")
        print(f"Total events: {len(events_df)}")
        print(f"Events by status:")
        print(events_df['event_status'].value_counts())
        print(f"\nEvents by region:")
        print(events_df['event_country'].value_counts())
        print(f"\nTop tournaments by prize pool:")
        top_prize = events_df.nlargest(10, 'prize_pool')[['event_name', 'prize_pool', 'event_status']]
        print(top_prize)
        
        return events_df
    else:
        return events


def collect_matches_data(save_csv=True):
    """Collect matches data"""
    print("=" * 60)
    print("COLLECTING VALORANT MATCHES DATA")
    print("=" * 60)
    
    collector = ValorantDataCollector()
    
    # Fetch all matches
    matches = collector.fetch_all_matches()
    
    if not matches:
        print("No matches data found")
        return None
    
    # Save to CSV if requested
    if save_csv:
        matches_df = collector.save_matches_to_csv(matches)
        
        # Display summary
        print("\nMATCHES DATA SUMMARY:")
        print(f"Total matches: {len(matches_df)}")
        print(f"Matches by status:")
        print(matches_df['status'].value_counts())
        print(f"\nMatches by tournament:")
        print(matches_df['tournament'].value_counts().head(10))
        print(f"\nUpcoming matches:")
        upcoming = matches_df[matches_df['status'] == 'Upcoming']
        if len(upcoming) > 0:
            print(f"Next {min(5, len(upcoming))} upcoming matches:")
            upcoming_display = upcoming[['team1_name', 'team2_name', 'tournament', 'time_until']].head(5)
            print(upcoming_display)
        
        return matches_df
    else:
        return matches


def collect_all_data_with_csv():
    """Collect all data and save to CSV files"""
    print("=" * 60)
    print("COLLECTING ALL VALORANT DATA WITH CSV EXPORT")
    print("=" * 60)
    
    collector = ValorantDataCollector()
    
    # Collect all data
    data = collector.collect_all_data(fetch_all_events=True, fetch_all_matches=True)
    
    # Save all data (JSON + CSV)
    collector.save_data(data, "valorant_complete_data")
    
    # Additionally save events and matches with special formatting
    if data['events']:
        events_df = collector.save_events_to_csv(data['events'], "valorant_events_detailed.csv")
    
    if data['matches']:
        matches_df = collector.save_matches_to_csv(data['matches'], "valorant_matches_detailed.csv")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED!")
    print("=" * 60)
    print("Files created:")
    print("- valorant_complete_data_*.json (all data in JSON format)")
    print("- valorant_complete_data_*.csv (all data in CSV format)")
    print("- valorant_events_detailed.csv (formatted events data)")
    print("- valorant_matches_detailed.csv (formatted matches data)")
    
    return data


def analyze_events_data(events_df):
    """Analyze events data and provide insights"""
    print("\n" + "=" * 60)
    print("EVENTS DATA ANALYSIS")
    print("=" * 60)
    
    # Tournament analysis
    print("Tournament Analysis:")
    print(f"Total tournaments: {len(events_df)}")
    print(f"Active tournaments: {len(events_df[events_df['event_status'] == 'ongoing'])}")
    print(f"Completed tournaments: {len(events_df[events_df['event_status'] == 'completed'])}")
    print(f"Upcoming tournaments: {len(events_df[events_df['event_status'] == 'upcoming'])}")
    
    # Prize pool analysis
    print(f"\nPrize Pool Analysis:")
    print(f"Total prize money: ${events_df['prize_pool'].sum():,.0f}")
    print(f"Average prize pool: ${events_df['prize_pool'].mean():,.0f}")
    print(f"Largest prize pool: ${events_df['prize_pool'].max():,.0f}")
    
    # Regional analysis
    print(f"\nRegional Distribution:")
    region_counts = events_df['event_country'].value_counts()
    for region, count in region_counts.head(10).items():
        print(f"  {region.upper()}: {count} tournaments")
    
    # Timeline analysis
    print(f"\nTournament Timeline (2025):")
    current_year = events_df[events_df['event_dates'].str.contains('2025', na=False)]
    if len(current_year) > 0:
        print(f"2025 tournaments: {len(current_year)}")
        print("Major 2025 tournaments:")
        major_2025 = current_year.nlargest(5, 'prize_pool')[['event_name', 'prize_pool', 'event_dates']]
        print(major_2025)


def analyze_matches_data(matches_df):
    """Analyze matches data and provide insights"""
    print("\n" + "=" * 60)
    print("MATCHES DATA ANALYSIS")
    print("=" * 60)
    
    # Match status analysis
    print("Match Status Analysis:")
    status_counts = matches_df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count} matches")
    
    # Tournament analysis
    print(f"\nTournament Analysis:")
    tournament_counts = matches_df['tournament'].value_counts()
    print("Top tournaments by match count:")
    for tournament, count in tournament_counts.head(10).items():
        print(f"  {tournament}: {count} matches")
    
    # Team analysis
    print(f"\nTeam Analysis:")
    all_teams = pd.concat([matches_df['team1_name'], matches_df['team2_name']]).dropna()
    team_counts = all_teams.value_counts()
    print("Most active teams:")
    for team, count in team_counts.head(10).items():
        print(f"  {team}: {count} matches")
    
    # Upcoming matches analysis
    upcoming = matches_df[matches_df['status'] == 'Upcoming']
    if len(upcoming) > 0:
        print(f"\nUpcoming Matches Analysis:")
        print(f"Total upcoming matches: {len(upcoming)}")
        print("Next 5 upcoming matches:")
        next_matches = upcoming[['team1_name', 'team2_name', 'tournament', 'time_until']].head(5)
        for _, match in next_matches.iterrows():
            print(f"  {match['team1_name']} vs {match['team2_name']} ({match['tournament']}) - {match['time_until']}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Collect Valorant events and matches data')
    parser.add_argument('--mode', choices=['events', 'matches', 'all'], default='all',
                       help='What data to collect')
    parser.add_argument('--status', choices=['all', 'ongoing', 'upcoming', 'completed'], 
                       default='all', help='Filter events by status')
    parser.add_argument('--region', choices=['all', 'na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc'],
                       default='all', help='Filter events by region')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV export')
    parser.add_argument('--analyze', action='store_true', help='Run data analysis after collection')
    
    args = parser.parse_args()
    
    print(f"Valorant Data Collection - Mode: {args.mode}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'events':
        events_df = collect_events_data(args.status, args.region, not args.no_csv)
        if args.analyze and events_df is not None:
            analyze_events_data(events_df)
    
    elif args.mode == 'matches':
        matches_df = collect_matches_data(not args.no_csv)
        if args.analyze and matches_df is not None:
            analyze_matches_data(matches_df)
    
    elif args.mode == 'all':
        data = collect_all_data_with_csv()
        if args.analyze:
            if 'events' in data and data['events']:
                events_df = pd.DataFrame(data['events'])
                events_df = events_df.rename(columns={
                    'id': 'event_id', 'name': 'event_name', 'status': 'event_status',
                    'prizepool': 'prize_pool', 'dates': 'event_dates', 
                    'country': 'event_country', 'img': 'event_image_url'
                })
                events_df['prize_pool'] = pd.to_numeric(events_df['prize_pool'], errors='coerce')
                analyze_events_data(events_df)
            
            if 'matches' in data and data['matches']:
                matches_df = pd.DataFrame(data['matches'])
                # Process matches data for analysis
                processed_matches = []
                for match in data['matches']:
                    teams = match.get('teams', [])
                    if len(teams) >= 2:
                        processed_match = {
                            'match_id': match.get('id'),
                            'team1_name': teams[0].get('name'),
                            'team2_name': teams[1].get('name'),
                            'status': match.get('status'),
                            'tournament': match.get('tournament'),
                            'time_until': match.get('in')
                        }
                        processed_matches.append(processed_match)
                
                if processed_matches:
                    matches_df = pd.DataFrame(processed_matches)
                    analyze_matches_data(matches_df)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
