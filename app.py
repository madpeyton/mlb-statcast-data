import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Load data
print("Loading data...")
url = 'https://raw.githubusercontent.com/madpeyton/mlb-statcast-data/main/2024_optimized_mapped.csv'

# Read CSV with proper handling
df_optimized = pd.read_csv(url, low_memory=False)

# Print debug info
print(f"Loaded {len(df_optimized)} rows")
print(f"Columns: {list(df_optimized.columns)[:10]}")  # First 10 columns

# Clean column names (remove any whitespace)
df_optimized.columns = df_optimized.columns.str.strip()

# Ensure datetime type
if 'game_date' in df_optimized.columns:
    if df_optimized['game_date'].dtype != 'datetime64[ns]':
        print("Converting game_date to datetime...")
        df_optimized['game_date'] = pd.to_datetime(df_optimized['game_date'], errors='coerce')
else:
    raise ValueError("game_date column not found!")

# Extract metadata
min_date = df_optimized['game_date'].min()
max_date = df_optimized['game_date'].max()
n_games = df_optimized['game_pk'].nunique()
n_pitches = len(df_optimized)
print(f"\nDate range: {min_date.date()} to {max_date.date()}")
print(f"Total games: {n_games:,}")
print(f"Total pitches: {n_pitches:,}")

# Extract unique values for filters
print("\nExtracting unique values for dropdowns...")
all_players = sorted(df_optimized['player_name_x'].dropna().unique())
all_teams = sorted(set(df_optimized['home_team'].unique()).union(set(df_optimized['away_team'].unique())))

# Get pitchers
df_optimized = df_optimized[~df_optimized["pitcher_display"].str.match(r"^\d+$")]
all_pitchers = sorted(set(df_optimized['pitcher_display'].unique()))
print(f"Unique batters: {len(all_players)}")
print(f"Unique pitchers: {len(all_pitchers)}")
print(f"Unique teams: {len(all_teams)}")

# Aggregate data
print("Aggregating player-game data...")
player_games = df_optimized.groupby(['player_name_x', 'game_date', 'game_pk']).agg({
    'launch_speed': 'mean',
    'launch_angle': 'mean',
    'hit_distance_sc': 'mean',
    'estimated_ba_using_speedangle': 'mean',
    'estimated_woba_using_speedangle': 'mean',
    'events': lambda x: x.notna().sum()
}).reset_index()
player_games.columns = ['player_name_x', 'game_date', 'game_pk', 'avg_exit_velo',
                        'avg_launch_angle', 'avg_distance', 'avg_xBA', 'avg_xwOBA',
                        'plate_appearances']

# Hard hit rate per game
hard_hits = df_optimized[df_optimized['launch_speed'] >= 95].groupby(
    ['player_name_x', 'game_date', 'game_pk']
).size().reset_index(name='hard_hits')
player_games = player_games.merge(hard_hits, on=['player_name_x', 'game_date', 'game_pk'], how='left')
player_games['hard_hits'] = player_games['hard_hits'].fillna(0)
player_games['hard_hit_rate'] = (player_games['hard_hits'] / player_games['plate_appearances'] * 100).fillna(0)
print(f"Player-game aggregations: {player_games.shape}")

# Team-game aggregations
print("Aggregating team-game data...")
team_games = df_optimized.groupby(['home_team', 'game_date', 'game_pk']).agg({
    'launch_speed': 'mean',
    'estimated_woba_using_speedangle': 'mean',
    'home_score': 'max',
    'events': lambda x: (x.isin(['single', 'double', 'triple', 'home_run'])).sum()
}).reset_index()
team_games.columns = ['team', 'game_date', 'game_pk', 'avg_exit_velo', 'avg_xwOBA', 'runs', 'hits']
print(f"Team-game aggregations: {team_games.shape}")

# Calculate league averages
league_avg_exit_velo = df_optimized['launch_speed'].mean()
league_avg_xwOBA = df_optimized['estimated_woba_using_speedangle'].mean()
league_avg_runs = team_games['runs'].mean()
print(f"League avg exit velocity: {league_avg_exit_velo:.1f} mph")
print(f"League avg xwOBA: {league_avg_xwOBA:.3f}")
print(f"League avg runs per game: {league_avg_runs:.2f}")

# Color schemes
COLORS = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#0d6efd',
    'success': '#198754',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'hot': '#d62728',
    'cold': '#1f77b4',
}

COLORSCALE_SEQUENTIAL = 'Viridis'
COLORSCALE_DIVERGING = 'RdBu'

# Helper functions
def create_empty_figure(message="No data available"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color='gray')
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )
    return fig

def calculate_percentile_rank(value, series):
    return stats.percentileofscore(series.dropna(), value, kind='rank')

def format_stat_card_value(value, metric_type='float'):
    if pd.isna(value):
        return "--"
    if metric_type == 'float':
        return f"{value:.1f}"
    elif metric_type == 'percent':
        return f"{value:.1f}%"
    elif metric_type == 'decimal':
        return f"{value:.3f}"
    else:
        return str(int(value))

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout functions
def create_global_controls():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("MLB Betting Insights Dashboard",
                       style={'textAlign': 'center', 'color': COLORS['primary'],
                              'marginTop': 20, 'marginBottom': 10}),
                html.P("Decision-support tool for performance analysis and betting insights",
                      style={'textAlign': 'center', 'fontSize': 16, 'color': COLORS['text']}),
                html.P([
                    f"Dataset: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} | ",
                    f"{n_games:,} games | {n_pitches:,} pitches"
                ], style={'textAlign': 'center', 'fontSize': 14, 'color': 'gray'}),
            ])
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H5("Global Controls", style={'marginBottom': 15}),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Start Date:", style={'fontWeight': 'bold'}),
                dcc.DatePickerSingle(
                    id='start-date-picker',
                    date=min_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format='YYYY-MM-DD'
                ),
            ], md=3),
            dbc.Col([
                html.Label("End Date:", style={'fontWeight': 'bold'}),
                dcc.DatePickerSingle(
                    id='end-date-picker',
                    date=max_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format='YYYY-MM-DD'
                ),
            ], md=3),
            dbc.Col([
                html.Label("Team:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='global-team-filter',
                    options=[{'label': 'All Teams', 'value': 'ALL'}] +
                            [{'label': team, 'value': team} for team in all_teams],
                    value='ALL',
                    clearable=False
                ),
            ], md=3)
        ], style={'marginBottom': 20}),
        html.Hr(),
    ], fluid=True)

def create_player_performance_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Select Player:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='player-selector',
                    options=[{'label': p, 'value': p} for p in all_players[:200]],
                    value=all_players[0] if all_players else None,
                    placeholder="Type to search players...",
                    searchable=True
                ),
            ], md=6)
        ], style={'marginTop': 20, 'marginBottom': 20}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Avg Exit Velocity", className="card-title"),
                        html.H3(id='player-exit-velo-card', children="--",
                               style={'color': COLORS['primary']}),
                        html.P(id='player-exit-velo-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Hard Hit Rate", className="card-title"),
                        html.H3(id='player-hard-hit-card', children="--",
                               style={'color': COLORS['success']}),
                        html.P(id='player-hard-hit-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Expected wOBA", className="card-title"),
                        html.H3(id='player-xwoba-card', children="--",
                               style={'color': COLORS['primary']}),
                        html.P(id='player-xwoba-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Games Analyzed", className="card-title"),
                        html.H3(id='player-games-card', children="--"),
                        html.P(id='player-games-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
        ], style={'marginBottom': 20}),
        dbc.Row([
            dbc.Col([
                dbc.Alert(id='player-trend-annotation', color="info",
                         children="Select a player to see performance insights")
            ])
        ], style={'marginBottom': 20}),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='player-exit-velo-trend',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
            dbc.Col([
                dcc.Graph(id='player-launch-angle-dist',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='player-spray-chart',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
            dbc.Col([
                dcc.Graph(id='player-outcomes-bar',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
        ], style={'marginTop': 20})
    ], fluid=True)

def create_team_performance_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Select Team:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='team-selector',
                    options=[{'label': team, 'value': team} for team in all_teams],
                    value=all_teams[0] if all_teams else None,
                    clearable=False
                ),
            ], md=4),
        ], style={'marginTop': 20, 'marginBottom': 20}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Avg Runs/Game", className="card-title"),
                        html.H3(id='team-runs-card', children="--",
                               style={'color': COLORS['primary']}),
                        html.P(id='team-runs-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Team Exit Velo", className="card-title"),
                        html.H3(id='team-exit-velo-card', children="--",
                               style={'color': COLORS['success']}),
                        html.P(id='team-exit-velo-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Expected wOBA", className="card-title"),
                        html.H3(id='team-xwoba-card', children="--",
                               style={'color': COLORS['primary']}),
                        html.P(id='team-xwoba-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Games Analyzed", className="card-title"),
                        html.H3(id='team-games-card', children="--"),
                        html.P(id='team-games-context', children="",
                              style={'fontSize': 12, 'color': 'gray'})
                    ])
                ], color="light")
            ], md=3),
        ], style={'marginBottom': 20}),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='team-runs-trend',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
            dbc.Col([
                dcc.Graph(id='team-exit-velo-vs-runs',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='team-xwoba-trend',
                         config={'displayModeBar': True, 'displaylogo': False})
            ], md=12),
        ], style={'marginTop': 20}),
        dbc.Row([
            dbc.Col([
                html.H5("League Rankings", style={'marginTop': 20, 'marginBottom': 10}),
                html.Div(id='team-rankings-table')
            ])
        ], style={'marginTop': 20})
    ], fluid=True)

def create_matchup_analyzer_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Select Batter:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='matchup-batter-selector',
                    options=[{'label': p, 'value': p} for p in all_players[:200]],
                    placeholder="Select batter...",
                    searchable=True
                ),
            ], md=6),
            dbc.Col([
                html.Label("Select Pitcher:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='matchup-pitcher-selector',
                    options=[{'label': p, 'value': p} for p in all_pitchers[:200]],
                    placeholder="Select pitcher...",
                    searchable=True
                ),
            ], md=6),
        ], style={'marginTop': 20, 'marginBottom': 20}),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="matchup-loading",
                    type="default",
                    color="#1f77b4",
                    fullscreen=False,
                    children=html.Div(
                        id='matchup-analyzer-content',
                        style={"minHeight": "400px"},
                        children=html.Div(
                            "Select a batter and pitcher to see matchup analysis",
                            style={"padding": "10px", "color": "#666"}
                        )
                    )
                ),
                md=12
            )
        ]),
    ], fluid=True)

# Main layout
app.layout = html.Div([
    create_global_controls(),
    dbc.Container([
        dcc.Tabs(id='main-tabs', value='tab-player-performance', children=[
            dcc.Tab(label='Player Performance', value='tab-player-performance'),
            dcc.Tab(label='Team Dashboard', value='tab-team-performance'),
            dcc.Tab(label='Matchup Analyzer', value='tab-matchup'),
        ]),
        html.Div(id='tab-content', style={'marginTop': 20}),
        html.Hr(style={'marginTop': 40}),
        html.P("Data Source: Baseball Savant Statcast",
              style={'textAlign': 'center', 'fontSize': 12, 'color': 'gray', 'marginBottom': 20})
    ], fluid=True)
], style={'backgroundColor': COLORS['background']})

# Tab rendering callback
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-player-performance':
        return create_player_performance_tab()
    elif tab == 'tab-team-performance':
        return create_team_performance_tab()
    elif tab == 'tab-matchup':
        return create_matchup_analyzer_tab()
    return html.Div("Tab not found")

# Player Performance Callbacks
@app.callback(
    Output('player-exit-velo-card', 'children'),
    Output('player-exit-velo-context', 'children'),
    Output('player-hard-hit-card', 'children'),
    Output('player-hard-hit-context', 'children'),
    Output('player-xwoba-card', 'children'),
    Output('player-xwoba-context', 'children'),
    Output('player-games-card', 'children'),
    Output('player-games-context', 'children'),
    Output('player-trend-annotation', 'children'),
    Output('player-exit-velo-trend', 'figure'),
    Output('player-launch-angle-dist', 'figure'),
    Output('player-spray-chart', 'figure'),
    Output('player-outcomes-bar', 'figure'),
    Input('player-selector', 'value'),
    Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
    Input('global-team-filter', 'value'),
)
def update_player_performance(player, start_date, end_date, team_filter):
    print(f"Updating player performance: {player}, {start_date} to {end_date}, team={team_filter}")

    if not player:
        print("No player selected")
        empty_fig = create_empty_figure("Select a player to view analysis")
        return ("--", "", "--", "", "--", "", "--", "",
                "Select a player to see performance insights",
                empty_fig, empty_fig, empty_fig, empty_fig)

    # Filter data
    mask = (
            (df_optimized['player_name_x'] == player) &
            (df_optimized['game_date'] >= start_date) &
            (df_optimized['game_date'] <= end_date)
    )

    if team_filter != 'ALL':
        mask &= ((df_optimized['home_team'] == team_filter) |
                 (df_optimized['away_team'] == team_filter))

    player_data = df_optimized[mask].copy()

    if len(player_data) == 0:
        print("No data for selection")
        empty_fig = create_empty_figure("No data available for this selection")
        return ("0", "", "0", "", "0", "", "0", "",
                "No data available for selected filters",
                empty_fig, empty_fig, empty_fig, empty_fig)

    print(f"Found {len(player_data)} plate appearances")

    # Calculate statistics
    avg_ev = player_data['launch_speed'].mean()
    hard_hits = (player_data['launch_speed'] >= 95).sum()
    total_hits = player_data['launch_speed'].notna().sum()
    hard_hit_rate = (hard_hits / total_hits * 100) if total_hits > 0 else 0
    avg_xwoba = player_data['estimated_woba_using_speedangle'].mean()
    num_games = player_data['game_pk'].nunique()

    # Calculate percentiles for context
    ev_percentile = calculate_percentile_rank(avg_ev, df_optimized['launch_speed'])
    xwoba_percentile = calculate_percentile_rank(avg_xwoba,
                                                 df_optimized['estimated_woba_using_speedangle'])

    # Create context strings
    ev_context = f"{ev_percentile:.0f}th percentile | League avg: {league_avg_exit_velo:.1f} mph"
    hh_context = f"{hard_hits}/{total_hits} batted balls"
    xwoba_context = f"{xwoba_percentile:.0f}th percentile | League avg: {league_avg_xwOBA:.3f}"
    games_context = f"From {start_date} to {end_date}"

    # Generate trend annotation
    # Compare recent 7 days to prior period
    recent_cutoff = pd.to_datetime(end_date) - timedelta(days=7)
    recent_data = player_data[player_data['game_date'] >= recent_cutoff]
    prior_data = player_data[player_data['game_date'] < recent_cutoff]

    if len(recent_data) > 0 and len(prior_data) > 0:
        recent_ev = recent_data['launch_speed'].mean()
        prior_ev = prior_data['launch_speed'].mean()
        ev_change = recent_ev - prior_ev
        ev_pct_change = (ev_change / prior_ev * 100) if prior_ev > 0 else 0

        recent_hh_rate = (recent_data['launch_speed'] >= 95).sum() / recent_data['launch_speed'].notna().sum() * 100
        prior_hh_rate = (prior_data['launch_speed'] >= 95).sum() / prior_data['launch_speed'].notna().sum() * 100
        hh_change = recent_hh_rate - prior_hh_rate

        if abs(ev_pct_change) > 2 or abs(hh_change) > 5:
            trend_text = f"Recent trend: {player}'s exit velocity {'increased' if ev_change > 0 else 'decreased'} by {abs(ev_pct_change):.1f}% in last 7 days. "
            trend_text += f"Hard-hit rate {'up' if hh_change > 0 else 'down'} {abs(hh_change):.1f} percentage points."
        else:
            trend_text = f"{player} showing consistent performance over the selected period."
    else:
        trend_text = f"Analyzing {player}'s performance over {num_games} games."

    # Figure 1: Exit velocity trend with rolling average
    daily_ev = player_data.groupby('game_date')['launch_speed'].mean().reset_index()
    daily_ev = daily_ev.sort_values('game_date')
    daily_ev['rolling_avg'] = daily_ev['launch_speed'].rolling(window=3, min_periods=1).mean()

    fig1 = go.Figure()

    # Actual values
    fig1.add_trace(go.Scatter(
        x=daily_ev['game_date'],
        y=daily_ev['launch_speed'],
        mode='markers',
        name='Actual',
        marker=dict(size=8, color=COLORS['primary'], opacity=0.6),
        hovertemplate='%{x|%Y-%m-%d}<br>Exit Velo: %{y:.1f} mph<extra></extra>'
    ))

    # Rolling average
    fig1.add_trace(go.Scatter(
        x=daily_ev['game_date'],
        y=daily_ev['rolling_avg'],
        mode='lines',
        name='3-Game Rolling Avg',
        line=dict(color=COLORS['success'], width=2)
    ))

    # League average reference line
    fig1.add_hline(
        y=league_avg_exit_velo,
        line_dash="dash",
        line_color='gray',
        annotation_text=f"League Avg: {league_avg_exit_velo:.1f}",
        annotation_position="right"
    )

    fig1.update_layout(
        title=f"{player} - Exit Velocity Trend",
        xaxis_title="Date",
        yaxis_title="Exit Velocity (mph)",
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white'
    )

    # Figure 2: Launch angle distribution with optimal zones
    launch_angles = player_data['launch_angle'].dropna()

    fig2 = go.Figure()

    fig2.add_trace(go.Histogram(
        x=launch_angles,
        nbinsx=30,
        name='Launch Angles',
        marker_color=COLORS['primary'],
        opacity=0.7
    ))

    # Add optimal range shading (8-32 degrees for power)
    fig2.add_vrect(
        x0=8, x1=32,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Optimal Range",
        annotation_position="top left"
    )

    fig2.update_layout(
        title=f"{player} - Launch Angle Distribution",
        xaxis_title="Launch Angle (degrees)",
        yaxis_title="Count",
        plot_bgcolor='white',
        bargap=0.05
    )

    # Figure 3: Spray chart (hit distribution)
    spray_data = player_data[player_data['hc_x'].notna() & player_data['hc_y'].notna()].copy()

    if len(spray_data) > 0:
        # Create outcome categories for color encoding
        def categorize_outcome(event):
            if event in ['home_run']:
                return 'Home Run'
            elif event in ['double', 'triple']:
                return 'Extra Base Hit'
            elif event in ['single']:
                return 'Single'
            else:
                return 'Out'

        spray_data['outcome_category'] = spray_data['events'].apply(categorize_outcome)

        fig3 = px.scatter(
            spray_data,
            x='hc_x',
            y='hc_y',
            color='outcome_category',
            color_discrete_map={
                'Home Run': COLORS['danger'],
                'Extra Base Hit': COLORS['warning'],
                'Single': COLORS['success'],
                'Out': 'lightgray'
            },
            title=f"{player} - Spray Chart",
            labels={'hc_x': 'Horizontal Position', 'hc_y': 'Vertical Position'},
            hover_data=['launch_speed', 'launch_angle', 'events']
        )

        fig3.update_layout(
            plot_bgcolor='#e6f4ea',  # Field green
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
    else:
        fig3 = create_empty_figure("No hit location data available")

    # Figure 4: Outcomes bar chart
    outcomes = player_data['events'].value_counts().head(10)

    fig4 = go.Figure()

    fig4.add_trace(go.Bar(
        x=outcomes.values,
        y=outcomes.index,
        orientation='h',
        marker_color=COLORS['primary'],
        text=outcomes.values,
        textposition='outside'
    ))

    fig4.update_layout(
        title=f"{player} - At-Bat Outcomes",
        xaxis_title="Count",
        yaxis_title="Outcome",
        plot_bgcolor='white',
        showlegend=False
    )

    return (
        format_stat_card_value(avg_ev, 'float'),
        ev_context,
        format_stat_card_value(hard_hit_rate, 'percent'),
        hh_context,
        format_stat_card_value(avg_xwoba, 'decimal'),
        xwoba_context,
        format_stat_card_value(num_games, 'int'),
        games_context,
        trend_text,
        fig1, fig2, fig3, fig4
    )

# Team Performance Callbacks
@app.callback(
    Output('team-runs-card', 'children'),
    Output('team-runs-context', 'children'),
    Output('team-exit-velo-card', 'children'),
    Output('team-exit-velo-context', 'children'),
    Output('team-xwoba-card', 'children'),
    Output('team-xwoba-context', 'children'),
    Output('team-games-card', 'children'),
    Output('team-games-context', 'children'),
    Output('team-runs-trend', 'figure'),
    Output('team-exit-velo-vs-runs', 'figure'),
    Output('team-xwoba-trend', 'figure'),
    Output('team-rankings-table', 'children'),
    Input('team-selector', 'value'),
    Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
)
def update_team_performance(team, start_date, end_date):
    print(f"Updating team performance: {team}")

    if not team:
        empty_fig = create_empty_figure("Select a team")
        empty_table = html.Div("Select a team to view rankings",
                               style={'padding': '10px', 'color': '#666'})
        return ("--", "", "--", "", "--", "", "--", "",
                empty_fig, empty_fig, empty_fig, empty_table)

    # Filter team data
    mask = (
            (team_games['team'] == team) &
            (team_games['game_date'] >= start_date) &
            (team_games['game_date'] <= end_date)
    )
    team_data = team_games[mask].copy()

    if len(team_data) == 0:
        empty_fig = create_empty_figure("No data available")
        empty_table = html.Div("No data available for rankings",
                               style={'padding': '10px', 'color': '#666'})
        return ("0", "", "0", "", "0", "", "0", "",
                empty_fig, empty_fig, empty_fig, empty_table)

    # Calculate stats
    avg_runs = team_data['runs'].mean()
    avg_ev = team_data['avg_exit_velo'].mean()
    avg_xwoba = team_data['avg_xwOBA'].mean()
    num_games = len(team_data)

    # Context
    runs_vs_league = avg_runs - league_avg_runs
    runs_context = f"{'+' if runs_vs_league > 0 else ''}{runs_vs_league:.2f} vs league avg"
    ev_context = f"League avg: {league_avg_exit_velo:.1f} mph"
    xwoba_context = f"League avg: {league_avg_xwOBA:.3f}"
    games_context = f"{num_games} games analyzed"

    # Figure 1: Runs trend
    team_data = team_data.sort_values('game_date')
    team_data['rolling_runs'] = team_data['runs'].rolling(window=5, min_periods=1).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=team_data['game_date'],
        y=team_data['runs'],
        mode='markers',
        name='Actual',
        marker=dict(size=6, color=COLORS['primary'], opacity=0.5)
    ))
    fig1.add_trace(go.Scatter(
        x=team_data['game_date'],
        y=team_data['rolling_runs'],
        mode='lines',
        name='5-Game Moving Avg',
        line=dict(color=COLORS['success'], width=2)
    ))
    fig1.add_hline(y=league_avg_runs, line_dash="dash", line_color='gray',
                   annotation_text="League Avg")
    fig1.update_layout(
        title=f"{team} - Runs per Game Trend",
        xaxis_title="Date",
        yaxis_title="Runs",
        plot_bgcolor='white'
    )

    # Figure 2: Exit velocity vs runs correlation
    fig2 = px.scatter(
        team_data,
        x='avg_exit_velo',
        y='runs',
        trendline='ols',
        title=f"{team} - Exit Velocity vs Runs",
        labels={'avg_exit_velo': 'Avg Exit Velocity (mph)', 'runs': 'Runs Scored'},
        hover_data=['game_date']
    )
    fig2.update_traces(marker=dict(size=8, color=COLORS['primary'], opacity=0.6))
    fig2.update_layout(plot_bgcolor='white')

    # Figure 3: xwOBA trend
    team_data['rolling_xwoba'] = team_data['avg_xwOBA'].rolling(window=5, min_periods=1).mean()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=team_data['game_date'],
        y=team_data['rolling_xwoba'],
        mode='lines',
        name='5-Game Rolling xwOBA',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy'
    ))
    fig3.add_hline(y=league_avg_xwOBA, line_dash="dash", line_color='gray',
                   annotation_text="League Avg")
    fig3.update_layout(
        title=f"{team} - Rolling Expected wOBA",
        xaxis_title="Date",
        yaxis_title="xwOBA",
        plot_bgcolor='white'
    )

    # Generate league rankings table
    mask_all = (
            (team_games['game_date'] >= start_date) &
            (team_games['game_date'] <= end_date)
    )
    league_data = team_games[mask_all].groupby('team').agg({
        'runs': 'mean',
        'avg_exit_velo': 'mean',
        'avg_xwOBA': 'mean',
        'game_pk': 'count'
    }).reset_index()

    league_data.columns = ['Team', 'Avg Runs', 'Avg Exit Velo', 'Avg xwOBA', 'Games']
    league_data = league_data.sort_values('Avg Runs', ascending=False).reset_index(drop=True)
    league_data['Rank'] = range(1, len(league_data) + 1)

    # Highlight the selected team
    def row_style(row):
        if row['Team'] == team:
            return {'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'}
        return {}

    # Create table
    table = dbc.Table.from_dataframe(
        league_data[['Rank', 'Team', 'Avg Runs', 'Avg Exit Velo', 'Avg xwOBA', 'Games']].round(2),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size='sm'
    )

    return (
        format_stat_card_value(avg_runs, 'float'),
        runs_context,
        format_stat_card_value(avg_ev, 'float'),
        ev_context,
        format_stat_card_value(avg_xwoba, 'decimal'),
        xwoba_context,
        format_stat_card_value(num_games, 'int'),
        games_context,
        fig1, fig2, fig3,
        table
    )

# Matchup Analyzer Callbacks
@app.callback(
    Output('matchup-analyzer-content', 'children'),
    Input('matchup-batter-selector', 'value'),
    Input('matchup-pitcher-selector', 'value'),
    Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
)
def update_matchup_analyzer(batter_value, pitcher_value, start_date, end_date):
    # Basic input checks
    if not batter_value or not pitcher_value or not start_date or not end_date:
        return html.Div(
            "Select a batter, pitcher, and valid date range to view matchup details.",
            style={"padding": "10px", "color": "#666"}
        )

    # Ensure valid dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date > end_date:
        return html.Div(
            "Start date cannot be after end date. Please adjust your selection.",
            style={"padding": "10px", "color": "#a00"}
        )

    # Filter data
    mask = (
            (df_optimized['pitcher_display'] == pitcher_value) &
            (df_optimized['game_date'] >= start_date) &
            (df_optimized['game_date'] <= end_date)
    )

    # Batter
    if 'player_name_x' in df_optimized.columns:
        mask &= (df_optimized['player_name_x'] == batter_value)
    elif 'batter' in df_optimized.columns:
        mask &= (df_optimized['batter'] == batter_value)

    matchup_df = df_optimized[mask]

    # If no data available
    if matchup_df.empty:
        return html.Div(
            "No pitches found for this batter–pitcher combination in the selected date range. "
            "Try a different matchup or expand the date window.",
            style={
                "padding": "12px",
                "border": "1px solid #ddd",
                "borderRadius": "6px",
                "marginTop": "10px",
                "color": "#555",
                "backgroundColor": "#fafafa"
            }
        )

    # Summary metrics
    total_pitches = len(matchup_df)

    if 'at_bat_number' in matchup_df.columns:
        pa_count = matchup_df['at_bat_number'].nunique()
    elif 'plate_appearance_id' in matchup_df.columns:
        pa_count = matchup_df['plate_appearance_id'].nunique()
    else:
        pa_count = None

    hits = None
    if 'events' in matchup_df.columns:
        hit_events = ['single', 'double', 'triple', 'home_run']
        hits = matchup_df['events'].isin(hit_events).sum()

    avg_ev = matchup_df['launch_speed'].mean() if 'launch_speed' in matchup_df.columns else None
    avg_xba = matchup_df['estimated_ba_using_speedangle'].mean() \
        if 'estimated_ba_using_speedangle' in matchup_df.columns else None

    # Summary cards helper
    def card(label, value):
        return html.Div(
            [
                html.Div(label, style={"fontSize": 11, "color": "#666"}),
                html.Div(value, style={"fontSize": 20, "fontWeight": "600"}),
            ],
            style={
                "padding": "8px 10px",
                "border": "1px solid #eee",
                "borderRadius": "6px",
                "backgroundColor": "#fafafa",
            },
        )

    summary_cols = [
        dbc.Col(card("Total Pitches", f"{total_pitches}"), md=2),
    ]

    if pa_count is not None:
        summary_cols.append(dbc.Col(card("Plate Appearances", f"{pa_count}"), md=2))

    if hits is not None:
        summary_cols.append(dbc.Col(card("Hits", f"{hits}"), md=2))

    if avg_ev is not None and not pd.isna(avg_ev):
        summary_cols.append(dbc.Col(card("Avg Exit Velo (mph)", f"{avg_ev:.1f}"), md=3))

    if avg_xba is not None and not pd.isna(avg_xba):
        summary_cols.append(dbc.Col(card("Avg xBA", f"{avg_xba:.3f}"), md=3))

    summary_row = dbc.Row(summary_cols, style={"marginBottom": "16px"})

    # Pitch location heatmap
    if {'plate_x', 'plate_z'}.issubset(matchup_df.columns):
        fig_heatmap = px.density_heatmap(
            matchup_df,
            x='plate_x',
            y='plate_z',
            nbinsx=9,
            nbinsy=9,
            color_continuous_scale='Viridis',
            labels={'plate_x': 'Horizontal Location', 'plate_z': 'Vertical Location'},
            title='Pitch Location Heatmap'
        )
        fig_heatmap.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=40, r=10, t=40, b=40),
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
        )
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.add_annotation(
            text="Pitch location data not available for this matchup.",
            x=0.5, y=0.5, showarrow=False,
        )
        fig_heatmap.update_layout(
            title='Pitch Location Heatmap',
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=40, r=10, t=40, b=40),
        )

    # Pitch mix bar chart
    if 'pitch_type' in matchup_df.columns:
        pitch_mix = (
            matchup_df
            .groupby('pitch_type')
            .size()
            .reset_index(name='pitches')
            .sort_values('pitches', ascending=False)
        )

        fig_pitch_mix = px.bar(
            pitch_mix,
            x='pitch_type',
            y='pitches',
            title='Pitch Mix (Pitches Thrown)',
            labels={'pitch_type': 'Pitch Type', 'pitches': 'Pitches'},
        )
        fig_pitch_mix.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=40, r=10, t=40, b=40),
        )
    else:
        fig_pitch_mix = go.Figure()
        fig_pitch_mix.add_annotation(
            text="Pitch type data not available for this matchup.",
            x=0.5, y=0.5, showarrow=False,
        )
        fig_pitch_mix.update_layout(
            title='Pitch Mix (Pitches Thrown)',
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(l=40, r=10, t=40, b=40),
        )

    # Final layout
    return html.Div([
        html.H5("Matchup Summary", style={"marginBottom": "8px"}),
        summary_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_heatmap), md=6),
            dbc.Col(dcc.Graph(figure=fig_pitch_mix), md=6),
        ])
    ])

# Run the server
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)