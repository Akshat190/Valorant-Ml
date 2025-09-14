"""
Demo script to collect Valorant VCT data from all regions
"""

from collect_all_regions import collect_all_regions_data, collect_specific_regions
from config import VCT_REGIONS, EVENT_STATUS
import pandas as pd


def demo_region_collection():
    """Demonstrate region-specific data collection"""
    print("=" * 80)
    print("VALORANT VCT ALL REGIONS DATA COLLECTION DEMO")
    print("=" * 80)
    
    # Show available regions
    print("\n🌍 Available VCT Regions:")
    for code, name in VCT_REGIONS.items():
        if code != 'all':
            print(f"  {code.upper()}: {name}")
    
    # Show available status filters
    print("\n📊 Available Event Status Filters:")
    for code, name in EVENT_STATUS.items():
        print(f"  {code}: {name}")
    
    print("\n" + "=" * 80)
    print("DEMO 1: Collect data from major regions only")
    print("=" * 80)
    
    # Collect from major regions
    major_regions = ['na', 'eu', 'kr', 'br', 'ap']
    print(f"Collecting data from: {', '.join([VCT_REGIONS[r] for r in major_regions])}")
    
    try:
        all_data, summaries = collect_specific_regions(major_regions, status='all', save_csv=True)
        
        print("\n✅ Major regions collection completed!")
        print("\nSummary:")
        for region, summary in summaries.items():
            if 'error' not in summary:
                print(f"  {summary['name']}: {summary['events']} events, {summary['matches']} matches")
            else:
                print(f"  {summary['name']}: Error - {summary['error']}")
                
    except Exception as e:
        print(f"❌ Error in major regions collection: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 2: Collect ongoing events from all regions")
    print("=" * 80)
    
    try:
        # Collect only ongoing events from all regions
        all_data, summaries = collect_all_regions_data(status='ongoing', save_csv=True)
        
        print("\n✅ Ongoing events collection completed!")
        print("\nOngoing events by region:")
        for region, summary in summaries.items():
            if 'error' not in summary and summary['events'] > 0:
                print(f"  {summary['name']}: {summary['events']} ongoing events")
                
    except Exception as e:
        print(f"❌ Error in ongoing events collection: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 3: Quick analysis of collected data")
    print("=" * 80)
    
    try:
        # Load and analyze the latest CSV files
        import glob
        import os
        
        # Find latest events file
        events_files = glob.glob("valorant_events_*.csv")
        if events_files:
            latest_events = max(events_files, key=os.path.getctime)
            events_df = pd.read_csv(latest_events)
            
            print(f"\n📊 Events Analysis (from {latest_events}):")
            print(f"Total events: {len(events_df)}")
            
            if 'event_status' in events_df.columns:
                print("\nEvents by status:")
                status_counts = events_df['event_status'].value_counts()
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")
            
            if 'event_country' in events_df.columns:
                print("\nEvents by region:")
                region_counts = events_df['event_country'].value_counts()
                for region, count in region_counts.head(10).items():
                    print(f"  {region.upper()}: {count}")
            
            if 'prize_pool' in events_df.columns:
                events_df['prize_pool'] = pd.to_numeric(events_df['prize_pool'], errors='coerce')
                print(f"\nPrize pool analysis:")
                print(f"  Total: ${events_df['prize_pool'].sum():,.0f}")
                print(f"  Average: ${events_df['prize_pool'].mean():,.0f}")
                print(f"  Largest: ${events_df['prize_pool'].max():,.0f}")
        
        # Find latest matches file
        matches_files = glob.glob("valorant_matches_*.csv")
        if matches_files:
            latest_matches = max(matches_files, key=os.path.getctime)
            matches_df = pd.read_csv(latest_matches)
            
            print(f"\n📊 Matches Analysis (from {latest_matches}):")
            print(f"Total matches: {len(matches_df)}")
            
            if 'status' in matches_df.columns:
                print("\nMatches by status:")
                status_counts = matches_df['status'].value_counts()
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")
            
            if 'tournament' in matches_df.columns:
                print("\nTop tournaments:")
                tournament_counts = matches_df['tournament'].value_counts()
                for tournament, count in tournament_counts.head(5).items():
                    print(f"  {tournament}: {count} matches")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED! 🎮")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run 'python collect_all_regions.py --regions all --status all --analyze' for full collection")
    print("2. Run 'python collect_all_regions.py --regions na eu kr --status ongoing' for specific regions")
    print("3. Use 'python analyze_csv_data.py --visualize --report' to analyze the data")
    print("4. Check the generated CSV files for detailed data")


if __name__ == "__main__":
    demo_region_collection()
