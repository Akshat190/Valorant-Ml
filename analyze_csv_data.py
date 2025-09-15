"""
Script to analyze Valorant CSV data files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
import argparse


def load_latest_csv_files():
    """Load the most recent CSV files from data/raw folder"""
    # Find the latest events file
    events_files = glob.glob("data/raw/valorant_events_*.csv") + glob.glob("data/raw/valorant_complete_data_events_*.csv")
    events_file = max(events_files, key=os.path.getctime) if events_files else None
    
    # Find the latest matches file
    matches_files = glob.glob("data/raw/valorant_matches_*.csv") + glob.glob("data/raw/valorant_complete_data_matches_*.csv")
    matches_file = max(matches_files, key=os.path.getctime) if matches_files else None
    
    # Find the latest results file
    results_files = glob.glob("data/raw/valorant_complete_data_results_*.csv")
    results_file = max(results_files, key=os.path.getctime) if results_files else None
    
    data = {}
    
    if events_file:
        print(f"Loading events data from: {events_file}")
        data['events'] = pd.read_csv(events_file)
    
    if matches_file:
        print(f"Loading matches data from: {matches_file}")
        data['matches'] = pd.read_csv(matches_file)
    
    if results_file:
        print(f"Loading results data from: {results_file}")
        data['results'] = pd.read_csv(results_file)
    
    return data


def analyze_events_data(events_df):
    """Comprehensive analysis of events data"""
    print("\n" + "=" * 60)
    print("EVENTS DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total events: {len(events_df)}")
    print(f"Columns: {list(events_df.columns)}")
    
    # Event status analysis
    if 'event_status' in events_df.columns:
        print(f"\nEvent Status Distribution:")
        status_counts = events_df['event_status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(events_df)*100:.1f}%)")
    
    # Prize pool analysis
    if 'prize_pool' in events_df.columns:
        print(f"\nPrize Pool Analysis:")
        print(f"  Total prize money: ${events_df['prize_pool'].sum():,.0f}")
        print(f"  Average prize pool: ${events_df['prize_pool'].mean():,.0f}")
        print(f"  Median prize pool: ${events_df['prize_pool'].median():,.0f}")
        print(f"  Largest prize pool: ${events_df['prize_pool'].max():,.0f}")
        
        # Top tournaments by prize pool
        print(f"\nTop 10 Tournaments by Prize Pool:")
        top_tournaments = events_df.nlargest(10, 'prize_pool')[['event_name', 'prize_pool', 'event_status', 'event_country']]
        for _, row in top_tournaments.iterrows():
            print(f"  {row['event_name']}: ${row['prize_pool']:,.0f} ({row['event_status']}, {row['event_country'].upper()})")
    
    # Regional analysis
    if 'event_country' in events_df.columns:
        print(f"\nRegional Distribution:")
        region_counts = events_df['event_country'].value_counts()
        for region, count in region_counts.head(10).items():
            print(f"  {region.upper()}: {count} tournaments ({count/len(events_df)*100:.1f}%)")
    
    # Timeline analysis
    if 'event_dates' in events_df.columns:
        print(f"\nTimeline Analysis:")
        # Extract year from dates
        events_df['year'] = events_df['event_dates'].str.extract(r'(\d{4})')
        year_counts = events_df['year'].value_counts().sort_index()
        print("Events by year:")
        for year, count in year_counts.items():
            if pd.notna(year):
                print(f"  {year}: {count} tournaments")
    
    return events_df


def analyze_matches_data(matches_df):
    """Comprehensive analysis of matches data"""
    print("\n" + "=" * 60)
    print("MATCHES DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total matches: {len(matches_df)}")
    print(f"Columns: {list(matches_df.columns)}")
    
    # Match status analysis
    if 'status' in matches_df.columns:
        print(f"\nMatch Status Distribution:")
        status_counts = matches_df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(matches_df)*100:.1f}%)")
    
    # Tournament analysis
    if 'tournament' in matches_df.columns:
        print(f"\nTournament Analysis:")
        tournament_counts = matches_df['tournament'].value_counts()
        print("Top 10 tournaments by match count:")
        for tournament, count in tournament_counts.head(10).items():
            print(f"  {tournament}: {count} matches")
    
    # Team analysis
    if 'team1_name' in matches_df.columns and 'team2_name' in matches_df.columns:
        print(f"\nTeam Analysis:")
        all_teams = pd.concat([matches_df['team1_name'], matches_df['team2_name']]).dropna()
        team_counts = all_teams.value_counts()
        print("Most active teams:")
        for team, count in team_counts.head(15).items():
            print(f"  {team}: {count} matches")
    
    # Regional analysis
    if 'team1_country' in matches_df.columns and 'team2_country' in matches_df.columns:
        print(f"\nRegional Analysis:")
        all_countries = pd.concat([matches_df['team1_country'], matches_df['team2_country']]).dropna()
        country_counts = all_countries.value_counts()
        print("Most active regions:")
        for country, count in country_counts.head(10).items():
            print(f"  {country.upper()}: {count} team appearances")
    
    # Upcoming matches analysis
    if 'status' in matches_df.columns:
        upcoming = matches_df[matches_df['status'] == 'Upcoming']
        if len(upcoming) > 0:
            print(f"\nUpcoming Matches Analysis:")
            print(f"Total upcoming matches: {len(upcoming)}")
            if 'time_until' in upcoming.columns:
                print("Next 10 upcoming matches:")
                next_matches = upcoming[['team1_name', 'team2_name', 'tournament', 'time_until']].head(10)
                for _, match in next_matches.iterrows():
                    print(f"  {match['team1_name']} vs {match['team2_name']} ({match['tournament']}) - {match['time_until']}")
    
    return matches_df


def analyze_results_data(results_df):
    """Comprehensive analysis of results data"""
    print("\n" + "=" * 60)
    print("RESULTS DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total match results: {len(results_df)}")
    print(f"Columns: {list(results_df.columns)}")
    
    # Tournament analysis
    if 'tournament' in results_df.columns:
        print(f"\nTournament Analysis:")
        tournament_counts = results_df['tournament'].value_counts()
        print("Top 10 tournaments by match count:")
        for tournament, count in tournament_counts.head(10).items():
            print(f"  {tournament}: {count} matches")
    
    # Team performance analysis
    if 'team1_name' in results_df.columns and 'team2_name' in results_df.columns:
        print(f"\nTeam Performance Analysis:")
        
        # Get all teams and their wins
        team_wins = {}
        team_matches = {}
        
        for _, match in results_df.iterrows():
            team1 = match['team1_name']
            team2 = match['team2_name']
            team1_won = match.get('team1_won', False)
            team2_won = match.get('team2_won', False)
            
            # Count matches
            team_matches[team1] = team_matches.get(team1, 0) + 1
            team_matches[team2] = team_matches.get(team2, 0) + 1
            
            # Count wins
            if team1_won:
                team_wins[team1] = team_wins.get(team1, 0) + 1
            if team2_won:
                team_wins[team2] = team_wins.get(team2, 0) + 1
        
        # Calculate win rates
        team_stats = []
        for team in team_matches:
            wins = team_wins.get(team, 0)
            matches = team_matches[team]
            win_rate = wins / matches if matches > 0 else 0
            team_stats.append({
                'team': team,
                'wins': wins,
                'matches': matches,
                'win_rate': win_rate
            })
        
        # Sort by win rate
        team_stats = sorted(team_stats, key=lambda x: x['win_rate'], reverse=True)
        
        print("Top 15 teams by win rate (min 5 matches):")
        for team_stat in team_stats[:15]:
            if team_stat['matches'] >= 5:
                print(f"  {team_stat['team']}: {team_stat['wins']}/{team_stat['matches']} ({team_stat['win_rate']:.1%})")
    
    return results_df


def create_visualizations(events_df, matches_df, results_df):
    """Create visualizations for the data"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Events by status
    if 'event_status' in events_df.columns:
        events_df['event_status'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Events by Status')
        axes[0, 0].set_xlabel('Status')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Prize pool distribution
    if 'prize_pool' in events_df.columns:
        events_df['prize_pool'].hist(bins=20, ax=axes[0, 1])
        axes[0, 1].set_title('Prize Pool Distribution')
        axes[0, 1].set_xlabel('Prize Pool ($)')
        axes[0, 1].set_ylabel('Frequency')
    
    # 3. Regional distribution
    if 'event_country' in events_df.columns:
        events_df['event_country'].value_counts().head(10).plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Events by Region (Top 10)')
        axes[0, 2].set_xlabel('Region')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Match status distribution
    if 'status' in matches_df.columns:
        matches_df['status'].value_counts().plot(kind='pie', ax=axes[1, 0])
        axes[1, 0].set_title('Match Status Distribution')
    
    # 5. Tournament activity
    if 'tournament' in matches_df.columns:
        matches_df['tournament'].value_counts().head(10).plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Tournaments by Match Count')
        axes[1, 1].set_xlabel('Tournament')
        axes[1, 1].set_ylabel('Match Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Team activity
    if 'team1_name' in matches_df.columns and 'team2_name' in matches_df.columns:
        all_teams = pd.concat([matches_df['team1_name'], matches_df['team2_name']]).dropna()
        all_teams.value_counts().head(10).plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Top 10 Most Active Teams')
        axes[1, 2].set_xlabel('Team')
        axes[1, 2].set_ylabel('Match Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('valorant_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'valorant_data_analysis.png'")


def generate_summary_report(events_df, matches_df, results_df):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 60)
    
    report = f"""
VALORANT ESPORTS DATA ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EVENTS DATA SUMMARY:
- Total Events: {len(events_df) if events_df is not None else 'N/A'}
- Active Events: {len(events_df[events_df['event_status'] == 'ongoing']) if events_df is not None and 'event_status' in events_df.columns else 'N/A'}
- Total Prize Pool: ${events_df['prize_pool'].sum():,.0f} if events_df is not None and 'prize_pool' in events_df.columns else 'N/A'

MATCHES DATA SUMMARY:
- Total Matches: {len(matches_df) if matches_df is not None else 'N/A'}
- Upcoming Matches: {len(matches_df[matches_df['status'] == 'Upcoming']) if matches_df is not None and 'status' in matches_df.columns else 'N/A'}
- Completed Matches: {len(matches_df[matches_df['status'] == 'Completed']) if matches_df is not None and 'status' in matches_df.columns else 'N/A'}

RESULTS DATA SUMMARY:
- Total Results: {len(results_df) if results_df is not None else 'N/A'}

KEY INSIGHTS:
- The data provides comprehensive coverage of Valorant esports
- Events span multiple regions and prize pools
- Match data includes both historical and upcoming matches
- Results data enables performance analysis and prediction modeling

RECOMMENDATIONS:
1. Use this data for machine learning model training
2. Regular updates will improve prediction accuracy
3. Focus on high-prize pool tournaments for better data quality
4. Consider regional performance differences in modeling
"""
    
    print(report)
    
    # Save report to file in results folder
    os.makedirs("data/results", exist_ok=True)
    with open('data/results/valorant_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Summary report saved as 'data/results/valorant_analysis_report.txt'")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze Valorant CSV data')
    parser.add_argument('--events-file', help='Path to events CSV file')
    parser.add_argument('--matches-file', help='Path to matches CSV file')
    parser.add_argument('--results-file', help='Path to results CSV file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    print("Valorant CSV Data Analysis")
    print("=" * 60)
    
    # Load data
    if args.events_file or args.matches_file or args.results_file:
        data = {}
        if args.events_file:
            data['events'] = pd.read_csv(args.events_file)
        if args.matches_file:
            data['matches'] = pd.read_csv(args.matches_file)
        if args.results_file:
            data['results'] = pd.read_csv(args.results_file)
    else:
        data = load_latest_csv_files()
    
    # Analyze data
    events_df = data.get('events')
    matches_df = data.get('matches')
    results_df = data.get('results')
    
    if events_df is not None:
        events_df = analyze_events_data(events_df)
    
    if matches_df is not None:
        matches_df = analyze_matches_data(matches_df)
    
    if results_df is not None:
        results_df = analyze_results_data(results_df)
    
    # Create visualizations
    if args.visualize:
        create_visualizations(events_df, matches_df, results_df)
    
    # Generate report
    if args.report:
        generate_summary_report(events_df, matches_df, results_df)
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
