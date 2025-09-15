"""
Configuration file for Valorant VCT Winner Prediction Model
"""

# API Configuration
VLR_API_BASE_URL = "https://vlr.orlandomm.net/api/v1"
VLR_ESPORTS_API_BASE_URL = "https://vlresports.vercel.app"
VLRGG_API_BASE_URL = "https://vlrggapi.vercel.app/api"

# API Endpoints
ENDPOINTS = {
    'results': f"{VLR_API_BASE_URL}/results",
    'teams': f"{VLR_ESPORTS_API_BASE_URL}/teams",
    'players': f"{VLR_ESPORTS_API_BASE_URL}/players",
    'events': f"{VLR_API_BASE_URL}/events",  # Use VLR API for events
    'matches': f"{VLR_API_BASE_URL}/matches",  # Use VLR API for matches
    'matches_all': f"{VLR_ESPORTS_API_BASE_URL}/matches/get-all-matches"  # Bulk matches endpoint
}

# VLRGG API Endpoints (extended coverage)
VLRGG_ENDPOINTS = {
    'news': f"{VLRGG_API_BASE_URL}/news",
    'stats': f"{VLRGG_API_BASE_URL}/stats",
    'rankings': f"{VLRGG_API_BASE_URL}/rankings",
    'match': f"{VLRGG_API_BASE_URL}/match",
    'events': f"{VLRGG_API_BASE_URL}/events",
    'health': f"{VLRGG_API_BASE_URL}/health"
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'feature_columns': [
        'team1_win_rate',
        'team2_win_rate',
        'team1_recent_form',
        'team2_recent_form',
        'team1_avg_score',
        'team2_avg_score',
        'team1_region_strength',
        'team2_region_strength',
        'head_to_head_advantage'
    ]
}

# Data Collection Configuration
DATA_CONFIG = {
    'max_results': 2000,  # Maximum number of results to fetch
    'min_matches_for_stats': 5,  # Minimum matches required for team statistics
    'recent_matches_window': 10,  # Number of recent matches to consider for form
    'max_events_pages': 20,  # Maximum pages to fetch for events (increased for better coverage)
    'max_matches_pages': 50  # Maximum pages to fetch for matches (increased for comprehensive data)
}

# VCT Regions Configuration
VCT_REGIONS = {
    'all': 'All Regions',
    'na': 'North America',
    'eu': 'Europe, Middle East & Africa',
    'br': 'Brazil',
    'ap': 'Asia Pacific',
    'kr': 'Korea',
    'ch': 'China',
    'jp': 'Japan',
    'lan': 'Latin America North',
    'las': 'Latin America South',
    'oce': 'Oceania',
    'mn': 'Middle East & North Africa',
    'gc': 'Game Changers'
}

# Event Status Configuration
EVENT_STATUS = {
    'all': 'All Events',
    'ongoing': 'Currently Running',
    'upcoming': 'Scheduled',
    'completed': 'Finished'
}
