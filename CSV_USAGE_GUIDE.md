# Valorant Data CSV Collection and Analysis Guide

This guide explains how to use the enhanced data collection system to fetch and analyze Valorant esports data in CSV format.

## 🚀 Quick Start

### 1. Collect All Data to CSV
```bash
# Collect all data and save to CSV files
python main.py --mode collect-csv

# Or use the specialized script
python collect_events_matches.py --mode all --analyze
```

### 2. Collect Specific Data Types
```bash
# Collect only events data
python collect_events_matches.py --mode events --status ongoing --region eu

# Collect only matches data
python collect_events_matches.py --mode matches

# Collect events with specific filters
python collect_events_matches.py --mode events --status completed --region na
```

### 3. Analyze CSV Data
```bash
# Analyze the latest CSV files
python main.py --mode analyze-csv

# Or use the analysis script directly
python analyze_csv_data.py --visualize --report
```

## 📊 Available Data Types

### Events Data (`valorant_events_*.csv`)
- **event_id**: Unique event identifier
- **event_name**: Tournament/event name
- **event_status**: ongoing, upcoming, completed
- **prize_pool**: Prize money in USD
- **event_dates**: Date range of the event
- **event_country**: Host country/region
- **event_image_url**: Event logo/image

### Matches Data (`valorant_matches_*.csv`)
- **match_id**: Unique match identifier
- **team1_name**: First team name
- **team1_country**: First team country
- **team1_score**: First team score (if completed)
- **team2_name**: Second team name
- **team2_country**: Second team country
- **team2_score**: Second team score (if completed)
- **status**: Upcoming, Completed, TBD
- **event**: Event stage/round
- **tournament**: Tournament name
- **time_until**: Time until match starts
- **timestamp**: Unix timestamp
- **utc_date**: UTC date string
- **match_datetime**: Parsed datetime

### Results Data (`valorant_complete_data_results_*.csv`)
- **match_id**: Unique match identifier
- **team1_name**: First team name
- **team1_country**: First team country
- **team1_score**: First team score
- **team1_won**: Boolean if team1 won
- **team2_name**: Second team name
- **team2_country**: Second team country
- **team2_score**: Second team score
- **team2_won**: Boolean if team2 won
- **winner**: Name of winning team
- **match_date**: Parsed match date
- **status**: Match status
- **event**: Event stage
- **tournament**: Tournament name

## 🔧 API Parameters

### Events API Filtering
- **status**: `all`, `ongoing`, `upcoming`, `completed`
- **region**: `all`, `na`, `eu`, `br`, `ap`, `kr`, `ch`, `jp`, `lan`, `las`, `oce`, `mn`, `gc`
- **page**: Page number for pagination

### Matches API
- **page**: Page number for pagination

## 📈 Data Analysis Features

### Events Analysis
- Tournament distribution by status and region
- Prize pool analysis and rankings
- Timeline analysis by year
- Regional performance insights

### Matches Analysis
- Match status distribution
- Tournament activity levels
- Team activity rankings
- Regional team participation
- Upcoming matches schedule

### Results Analysis
- Team performance statistics
- Win rate calculations
- Tournament performance tracking
- Historical match analysis

## 🎯 Usage Examples

### Example 1: Collect Current VCT Data
```bash
# Get all ongoing and upcoming VCT events
python collect_events_matches.py --mode events --status ongoing --analyze

# Get all upcoming matches
python collect_events_matches.py --mode matches --analyze
```

### Example 2: Regional Analysis
```bash
# Analyze North American events
python collect_events_matches.py --mode events --region na --analyze

# Analyze European matches
python collect_events_matches.py --mode matches --analyze
```

### Example 3: Historical Data
```bash
# Get all completed events for historical analysis
python collect_events_matches.py --mode events --status completed --analyze
```

### Example 4: Custom Analysis
```bash
# Analyze specific CSV files
python analyze_csv_data.py --events-file valorant_events_20240914.csv --visualize --report
```

## 📁 File Outputs

### Generated Files
- `valorant_events_YYYYMMDD_HHMMSS.csv` - Events data
- `valorant_matches_YYYYMMDD_HHMMSS.csv` - Matches data
- `valorant_complete_data_*.csv` - All data types
- `valorant_events_detailed.csv` - Formatted events
- `valorant_matches_detailed.csv` - Formatted matches
- `valorant_data_analysis.png` - Visualization charts
- `valorant_analysis_report.txt` - Summary report

### JSON Files (Backup)
- `valorant_data_events_*.json` - Events in JSON format
- `valorant_data_matches_*.json` - Matches in JSON format
- `valorant_data_results_*.json` - Results in JSON format

## 🔄 Data Updates

### Regular Updates
```bash
# Update data daily
python main.py --mode collect-csv

# Update only events
python collect_events_matches.py --mode events

# Update only matches
python collect_events_matches.py --mode matches
```

### Scheduled Updates
You can set up cron jobs or scheduled tasks to run:
```bash
# Daily at 6 AM
0 6 * * * cd /path/to/valorant-ml && python main.py --mode collect-csv

# Every 4 hours
0 */4 * * * cd /path/to/valorant-ml && python collect_events_matches.py --mode matches
```

## 🎮 Integration with ML Model

### Using CSV Data for Training
```python
import pandas as pd
from data_preprocessor import ValorantDataPreprocessor

# Load CSV data
results_df = pd.read_csv('valorant_complete_data_results_20240914.csv')

# Convert to format expected by preprocessor
raw_data = {'results': results_df.to_dict('records')}

# Preprocess for ML
preprocessor = ValorantDataPreprocessor()
df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
```

### Using CSV Data for Predictions
```python
import pandas as pd
from prediction_interface import ValorantPredictionInterface

# Load model
predictor = ValorantPredictionInterface('valorant_model_random_forest')

# Load team stats from CSV
team_stats_df = pd.read_csv('processed_valorant_data_team_stats.csv', index_col=0)
team_stats = team_stats_df.to_dict('index')

# Make prediction
prediction = predictor.predict_match('G2 Esports', 'Team Heretics', 'us', 'eu')
```

## 🚨 Important Notes

1. **API Rate Limits**: The system includes delays to respect API rate limits
2. **Data Freshness**: Events and matches data changes frequently
3. **File Management**: Old CSV files are not automatically cleaned up
4. **Error Handling**: The system handles API errors gracefully
5. **Data Validation**: Always check data quality before using for ML

## 🔍 Troubleshooting

### Common Issues
1. **No data returned**: Check API status and internet connection
2. **Empty CSV files**: Verify API responses and data structure
3. **Missing columns**: Update data collector if API structure changes
4. **Encoding issues**: CSV files use UTF-8 encoding

### Debug Mode
```bash
# Run with verbose output
python collect_events_matches.py --mode all --analyze -v
```

## 📞 Support

For issues or questions:
1. Check the generated log files
2. Verify API endpoints are accessible
3. Review the data structure in JSON files
4. Check the analysis report for insights
