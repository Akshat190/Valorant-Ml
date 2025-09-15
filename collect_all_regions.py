"""
Comprehensive script to collect Valorant VCT data from all regions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import time
from data_collector import ValorantDataCollector
from config import VCT_REGIONS, EVENT_STATUS


def collect_region_data(region, status="all", save_csv=True):
    """Collect data for a specific region"""
    print(f"\n{'='*60}")
    print(f"COLLECTING DATA FOR {VCT_REGIONS[region].upper()}")
    print(f"{'='*60}")
    print(f"Region: {region} ({VCT_REGIONS[region]})")
    print(f"Status: {status} ({EVENT_STATUS[status]})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    collector = ValorantDataCollector()
    
    # Collect events for this region
    print(f"\nFetching events for {region}...")
    events = collector.fetch_all_events(status=status, region=region)
    
    # Collect matches (no region filter available for matches API)
    print(f"\nFetching matches...")
    matches = collector.fetch_all_matches()
    
    # Collect results
    print(f"\nFetching results...")
    results = collector.fetch_results(limit=1000)
    
    # Skip teams and players (APIs returning 404)
    print(f"\nSkipping teams and players (APIs not available)...")
    teams = []
    players = []
    
    # Prepare data
    data = {
        'events': events,
        'matches': matches,
        'results': results,
        'teams': teams,
        'players': players
    }
    
    # Save data
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"valorant_{region}_{status}_{timestamp}"
        
        # Save all data
        collector.save_data(data, prefix)
        
        # Save specialized CSV files
        if events:
            events_df = collector.save_events_to_csv(events, f"valorant_events_{region}_{status}_{timestamp}.csv")
            print(f"✓ Events CSV saved: {len(events_df)} events")
        
        if matches:
            matches_df = collector.save_matches_to_csv(matches, f"valorant_matches_{region}_{status}_{timestamp}.csv")
            print(f"✓ Matches CSV saved: {len(matches_df)} matches")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETED FOR {VCT_REGIONS[region].upper()}")
    print(f"{'='*60}")
    print(f"Events: {len(events)}")
    print(f"Matches: {len(matches)}")
    print(f"Results: {len(results)}")
    print(f"Teams: {len(teams)}")
    print(f"Players: {len(players)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return data


def collect_all_regions_data(status="all", save_csv=True):
    """Collect data from all VCT regions"""
    print("=" * 80)
    print("COMPREHENSIVE VALORANT VCT DATA COLLECTION - ALL REGIONS")
    print("=" * 80)
    print(f"Status filter: {status} ({EVENT_STATUS[status]})")
    print(f"Regions: {len(VCT_REGIONS)} regions")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_data = {}
    region_summaries = {}
    
    # Collect data for each region
    for region_code, region_name in VCT_REGIONS.items():
        if region_code == 'all':
            continue  # Skip 'all' as it's not a specific region
            
        try:
            print(f"\n🔄 Processing {region_name} ({region_code})...")
            region_data = collect_region_data(region_code, status, save_csv)
            all_data[region_code] = region_data
            region_summaries[region_code] = {
                'name': region_name,
                'events': len(region_data['events']),
                'matches': len(region_data['matches']),
                'results': len(region_data['results']),
                'teams': len(region_data['teams']),
                'players': len(region_data['players'])
            }
            
            # Add delay between regions to be respectful to the API
            print(f"⏳ Waiting 2 seconds before next region...")
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error collecting data for {region_name}: {e}")
            region_summaries[region_code] = {
                'name': region_name,
                'error': str(e),
                'events': 0,
                'matches': 0,
                'results': 0,
                'teams': 0,
                'players': 0
            }
    
    # Generate comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COLLECTION SUMMARY")
    print(f"{'='*80}")
    
    total_events = 0
    total_matches = 0
    total_results = 0
    total_teams = 0
    total_players = 0
    
    print(f"{'Region':<20} {'Events':<8} {'Matches':<8} {'Results':<8} {'Teams':<8} {'Players':<8} {'Status'}")
    print("-" * 80)
    
    for region_code, summary in region_summaries.items():
        if 'error' in summary:
            status_text = "❌ Error"
        else:
            status_text = "✅ Success"
            total_events += summary['events']
            total_matches += summary['matches']
            total_results += summary['results']
            total_teams += summary['teams']
            total_players += summary['players']
        
        print(f"{summary['name']:<20} {summary['events']:<8} {summary['matches']:<8} {summary['results']:<8} {summary['teams']:<8} {summary['players']:<8} {status_text}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_events:<8} {total_matches:<8} {total_results:<8} {total_teams:<8} {total_players:<8}")
    print(f"{'='*80}")
    print(f"Collection completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save comprehensive summary
    if save_csv:
        summary_df = pd.DataFrame.from_dict(region_summaries, orient='index')
        summary_df.to_csv(f"valorant_all_regions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"✓ Summary saved to CSV")
    
    return all_data, region_summaries


def collect_specific_regions(regions, status="all", save_csv=True):
    """Collect data for specific regions only"""
    print("=" * 80)
    print("VALORANT VCT DATA COLLECTION - SPECIFIC REGIONS")
    print("=" * 80)
    print(f"Regions: {', '.join([VCT_REGIONS[r] for r in regions])}")
    print(f"Status: {status} ({EVENT_STATUS[status]})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_data = {}
    region_summaries = {}
    
    for region in regions:
        if region not in VCT_REGIONS:
            print(f"❌ Unknown region: {region}")
            continue
            
        try:
            print(f"\n🔄 Processing {VCT_REGIONS[region]} ({region})...")
            region_data = collect_region_data(region, status, save_csv)
            all_data[region] = region_data
            region_summaries[region] = {
                'name': VCT_REGIONS[region],
                'events': len(region_data['events']),
                'matches': len(region_data['matches']),
                'results': len(region_data['results']),
                'teams': len(region_data['teams']),
                'players': len(region_data['players'])
            }
            
            time.sleep(2)  # Delay between regions
            
        except Exception as e:
            print(f"❌ Error collecting data for {VCT_REGIONS[region]}: {e}")
            region_summaries[region] = {
                'name': VCT_REGIONS[region],
                'error': str(e),
                'events': 0,
                'matches': 0,
                'results': 0,
                'teams': 0,
                'players': 0
            }
    
    return all_data, region_summaries


def analyze_regional_data(all_data, region_summaries):
    """Analyze the collected regional data"""
    print(f"\n{'='*80}")
    print("REGIONAL DATA ANALYSIS")
    print(f"{'='*80}")
    
    # Combine all events data
    all_events = []
    for region_data in all_data.values():
        all_events.extend(region_data['events'])
    
    if all_events:
        events_df = pd.DataFrame(all_events)
        print(f"\n📊 EVENTS ANALYSIS:")
        print(f"Total events across all regions: {len(events_df)}")
        
        if 'country' in events_df.columns:
            print(f"\nEvents by region:")
            region_counts = events_df['country'].value_counts()
            for region, count in region_counts.head(10).items():
                print(f"  {region.upper()}: {count} events")
        
        if 'status' in events_df.columns:
            print(f"\nEvents by status:")
            status_counts = events_df['status'].value_counts()
            for status, count in status_counts.items():
                print(f"  {status}: {count} events")
        
        if 'prizepool' in events_df.columns:
            events_df['prizepool'] = pd.to_numeric(events_df['prizepool'], errors='coerce')
            print(f"\nPrize pool analysis:")
            print(f"  Total prize money: ${events_df['prizepool'].sum():,.0f}")
            print(f"  Average prize pool: ${events_df['prizepool'].mean():,.0f}")
            print(f"  Largest prize pool: ${events_df['prizepool'].max():,.0f}")
    
    # Combine all matches data
    all_matches = []
    for region_data in all_data.values():
        all_matches.extend(region_data['matches'])
    
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        print(f"\n📊 MATCHES ANALYSIS:")
        print(f"Total matches across all regions: {len(matches_df)}")
        
        if 'tournament' in matches_df.columns:
            print(f"\nTop tournaments by match count:")
            tournament_counts = matches_df['tournament'].value_counts()
            for tournament, count in tournament_counts.head(10).items():
                print(f"  {tournament}: {count} matches")
    
    # Regional comparison
    print(f"\n📊 REGIONAL COMPARISON:")
    successful_regions = {k: v for k, v in region_summaries.items() if 'error' not in v}
    
    if successful_regions:
        print(f"Most active regions by events:")
        region_events = [(v['name'], v['events']) for v in successful_regions.values()]
        region_events.sort(key=lambda x: x[1], reverse=True)
        for region_name, count in region_events[:5]:
            print(f"  {region_name}: {count} events")
    
    print(f"\n{'='*80}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Collect Valorant VCT data from all regions')
    parser.add_argument('--regions', nargs='+', 
                       choices=list(VCT_REGIONS.keys()),
                       default=['all'],
                       help='Regions to collect data from')
    parser.add_argument('--status', choices=list(EVENT_STATUS.keys()),
                       default='all',
                       help='Filter events by status')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export')
    parser.add_argument('--analyze', action='store_true',
                       help='Run data analysis after collection')
    parser.add_argument('--delay', type=int, default=2,
                       help='Delay between region requests (seconds)')
    
    args = parser.parse_args()
    
    print("Valorant VCT All Regions Data Collection")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if 'all' in args.regions:
        # Collect from all regions
        all_data, region_summaries = collect_all_regions_data(args.status, not args.no_csv)
    else:
        # Collect from specific regions
        all_data, region_summaries = collect_specific_regions(args.regions, args.status, not args.no_csv)
    
    if args.analyze:
        analyze_regional_data(all_data, region_summaries)
    
    print(f"\n🎮 Collection completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
