#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNPS-10: Comprehensive National Power Assessment System
Open Source Research Platform - Main Web Interface

This module provides the main web interface for the CNPS-10 system,
offering national power assessment and comparison capabilities.

Features:
- Multi-year analysis (2000-2050)
- Year selection and comparison functionality
- Statistical analysis and visualizations
- Interactive charts and reports
- Country comparisons

Author: Open Source Research Project
Version: 1.0.0
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page settings
st.set_page_config(
    page_title="CNPS-10: Academic Research Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cnps-10/help',
        'Report a Bug': 'https://github.com/cnps-10/issues',
        'About': "CNPS-10: Comprehensive National Power Assessment System v1.0.0"
    }
)

# Academic styling CSS
st.markdown("""
<style>
.main-header {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #1e3a8a;
    margin-bottom: 10px;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 10px;
}

.sub-header {
    font-size: 20px;
    text-align: center;
    color: #4b5563;
    margin-bottom: 30px;
    font-style: italic;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}

.info-box {
    background-color: #f8fafc;
    border-left: 5px solid #3b82f6;
    padding: 15px;
    margin: 20px 0;
    border-radius: 5px;
}

.warning-box {
    background-color: #fef3cd;
    border-left: 5px solid #fbbf24;
    padding: 15px;
    margin: 20px 0;
    border-radius: 5px;
}

.success-box {
    background-color: #dcfce7;
    border-left: 5px solid #22c55e;
    padding: 15px;
    margin: 20px 0;
    border-radius: 5px;
}

.sidebar .sidebar-content {
    background-color: #f1f5f9;
}

.year-selector {
    background-color: #e2e8f0;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

.country-rank {
    padding: 8px 15px;
    margin: 5px 0;
    border-radius: 20px;
    font-weight: bold;
}

.rank-1 { background: linear-gradient(90deg, #ffd700, #ffed4e); color: #1a1a1a; }
.rank-2 { background: linear-gradient(90deg, #c0c0c0, #e5e5e5); color: #1a1a1a; }
.rank-3 { background: linear-gradient(90deg, #cd7f32, #daa520); color: white; }
.rank-other { background-color: #e2e8f0; color: #374151; }
</style>
""", unsafe_allow_html=True)

class CNPS10System:
    """
    CNPS-10 Academic Research System
    
    Main class for handling data loading, processing, and visualization
    for the Comprehensive National Power Assessment System.
    """
    
    def __init__(self):
        """Initialize the CNPS-10 system and load required data."""
        self.load_data()
        
    def load_data(self):
        """
        Load multi-year data and latest rankings.
        
        Priority order for data loading:
        1. Official-based 172 countries dataset
        2. Latest objective rankings
        3. Fallback to general rankings
        """
        try:
            # Load multi-year data (primary dataset)
            data_file = 'outputs/data/cnps10_official_based_172countries_20250903_022541.csv'
            if os.path.exists(data_file):
                self.multi_year_data = pd.read_csv(data_file)
                print(f"✅ Loaded multi-year data: {len(self.multi_year_data)} records")
            else:
                st.error(f"❌ Multi-year data file not found: {data_file}")
                self.multi_year_data = None
                
            # Load latest rankings (fallback options)
            ranking_files = [
                'outputs/rankings/cnps10_objective_ranking_2025.csv',
                'outputs/cnps10_ranking_2025.csv'
            ]
            
            self.latest_data = None
            for file_path in ranking_files:
                if os.path.exists(file_path):
                    self.latest_data = pd.read_csv(file_path)
                    print(f"✅ Loaded latest rankings: {len(self.latest_data)} countries")
                    break
                    
            if self.latest_data is None:
                st.error("❌ No ranking data files found")
                
            # Standardize column names for consistency
            self._standardize_columns()
            
            # Memory optimization
            self._optimize_memory()
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            print(f"Error details: {e}")
            
    def _standardize_columns(self):
        """
        Standardize column names across different datasets.
        
        Ensures consistent naming convention:
        - cnps10_score: Main CNPS-10 score
        - power_tier: Power classification tier
        - country: Country name
        """
        if self.multi_year_data is not None:
            # Check and rename columns as needed
            column_mapping = {
                'CNPS10_score': 'cnps10_score',
                'Country': 'country',
                'Power_Tier': 'power_tier'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in self.multi_year_data.columns:
                    self.multi_year_data.rename(columns={old_name: new_name}, inplace=True)
                    
        if self.latest_data is not None:
            # Apply same mapping to latest data
            column_mapping = {
                'CNPS10_score': 'cnps10_score',
                'Country': 'country',
                'Power_Tier': 'power_tier'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in self.latest_data.columns:
                    self.latest_data.rename(columns={old_name: new_name}, inplace=True)
    
    def _optimize_memory(self):
        """
        Optimize memory usage by adjusting data types with improved precision.
        
        Converts appropriate columns to more efficient data types
        while maintaining high precision for calculations.
        """
        if self.multi_year_data is not None:
            # Convert year to int16 (saves memory)
            if 'year' in self.multi_year_data.columns:
                self.multi_year_data['year'] = self.multi_year_data['year'].astype('int16')
                
            # Use float64 for scores to maintain high precision (not float32)
            score_cols = [col for col in self.multi_year_data.columns if 'score' in col.lower()]
            for col in score_cols:
                self.multi_year_data[col] = self.multi_year_data[col].astype('float64')
                
            # Convert categorical columns to category type
            categorical_cols = ['country', 'power_tier']
            for col in categorical_cols:
                if col in self.multi_year_data.columns:
                    self.multi_year_data[col] = self.multi_year_data[col].astype('category')
        
        # Apply same optimizations to latest data
        if self.latest_data is not None:
            score_cols = [col for col in self.latest_data.columns if 'score' in col.lower()]
            for col in score_cols:
                self.latest_data[col] = self.latest_data[col].astype('float64')  # Use float64 for precision
                
            categorical_cols = ['country', 'power_tier']
            for col in categorical_cols:
                if col in self.latest_data.columns:
                    self.latest_data[col] = self.latest_data[col].astype('category')
        
        # Force garbage collection
        gc.collect()
    
    def get_available_years(self):
        """
        Get list of available years in the dataset.
        
        Returns:
            list: Sorted list of available years
        """
        if self.multi_year_data is not None and 'year' in self.multi_year_data.columns:
            return sorted(self.multi_year_data['year'].unique())
        return [2025]  # Fallback to current year
    
    def get_countries_by_year(self, year):
        """
        Get countries data for a specific year.
        
        Args:
            year (int): Year to filter data for
            
        Returns:
            pandas.DataFrame: Filtered data for the specified year
        """
        if self.multi_year_data is not None and 'year' in self.multi_year_data.columns:
            return self.multi_year_data[self.multi_year_data['year'] == year].copy()
        else:
            # Fallback to latest data
            return self.latest_data.copy() if self.latest_data is not None else pd.DataFrame()
    
    def get_top_countries(self, year, top_n=10):
        """
        Get top N countries for a specific year.
        
        Args:
            year (int): Year to analyze
            top_n (int): Number of top countries to return
            
        Returns:
            pandas.DataFrame: Top countries sorted by CNPS-10 score
        """
        data = self.get_countries_by_year(year)
        if not data.empty and 'cnps10_score' in data.columns:
            return data.nlargest(top_n, 'cnps10_score')
        return pd.DataFrame()
    
    def create_trend_chart(self, countries, start_year, end_year):
        """
        Create trend chart for selected countries.
        
        Args:
            countries (list): List of country names
            start_year (int): Start year for analysis
            end_year (int): End year for analysis
            
        Returns:
            plotly.graph_objects.Figure: Interactive trend chart
        """
        if self.multi_year_data is None:
            return go.Figure()
            
        # Filter data
        mask = (
            (self.multi_year_data['year'] >= start_year) & 
            (self.multi_year_data['year'] <= end_year) &
            (self.multi_year_data['country'].isin(countries))
        )
        filtered_data = self.multi_year_data[mask]
        
        # Create line chart
        fig = px.line(
            filtered_data,
            x='year',
            y='cnps10_score',
            color='country',
            title=f'CNPS-10 Score Trends ({start_year}-{end_year})',
            labels={
                'cnps10_score': 'CNPS-10 Score (%)',
                'year': 'Year',
                'country': 'Country'
            }
        )
        
        # Improve layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            height=500,
            hovermode='x unified'
        )
        
        # Format y-axis as percentage
        fig.update_yaxes(tickformat='.1%')
        
        return fig
    
    def create_ranking_chart(self, year, top_n=15):
        """
        Create horizontal bar chart for country rankings.
        
        Args:
            year (int): Year to analyze
            top_n (int): Number of countries to display
            
        Returns:
            plotly.graph_objects.Figure: Horizontal bar chart
        """
        data = self.get_top_countries(year, top_n)
        
        if data.empty:
            return go.Figure()
        
        # Sort in ascending order for proper horizontal bar display
        data_sorted = data.sort_values('cnps10_score', ascending=True)
        
        # Create color scale based on ranking
        colors = ['#1f77b4'] * len(data_sorted)
        if len(colors) >= 1: colors[-1] = '#FFD700'  # Gold for #1
        if len(colors) >= 2: colors[-2] = '#C0C0C0'  # Silver for #2
        if len(colors) >= 3: colors[-3] = '#CD7F32'  # Bronze for #3
        
        fig = go.Figure(data=[
            go.Bar(
                y=data_sorted['country'],
                x=data_sorted['cnps10_score'],
                orientation='h',
                marker_color=colors,
                text=data_sorted['cnps10_score'].apply(lambda x: f'{x:.1%}'),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Countries - CNPS-10 Rankings ({year})',
            title_font_size=20,
            xaxis_title='CNPS-10 Score (%)',
            yaxis_title='Country',
            height=max(400, len(data_sorted) * 30),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # Format x-axis as percentage
        fig.update_xaxes(tickformat='.1%')
        
        return fig
    
    def generate_statistics_summary(self, year):
        """
        Generate comprehensive statistics summary for a given year.
        
        Args:
            year (int): Year to analyze
            
        Returns:
            dict: Dictionary containing various statistical measures
        """
        data = self.get_countries_by_year(year)
        
        if data.empty or 'cnps10_score' not in data.columns:
            return {}
        
        scores = data['cnps10_score']
        
        stats = {
            'total_countries': len(data),
            'mean_score': scores.mean(),
            'median_score': scores.median(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'q1_score': scores.quantile(0.25),
            'q3_score': scores.quantile(0.75),
            'skewness': scores.skew(),
            'kurtosis': scores.kurtosis()
        }
        
        return stats
    
    def calculate_power_transitions(self, start_year, end_year):
        """
        Calculate power transitions between countries over time period.
        
        Args:
            start_year (int): Start year for analysis
            end_year (int): End year for analysis
            
        Returns:
            dict: Power transition analysis results
        """
        if self.multi_year_data is None:
            return {}
        
        # Get start and end year data
        start_data = self.get_countries_by_year(start_year)
        end_data = self.get_countries_by_year(end_year)
        
        if start_data.empty or end_data.empty:
            return {}
        
        # Calculate rank changes
        start_ranks = start_data.set_index('country')['cnps10_score'].rank(ascending=False)
        end_ranks = end_data.set_index('country')['cnps10_score'].rank(ascending=False)
        
        # Find countries present in both years
        common_countries = start_ranks.index.intersection(end_ranks.index)
        
        transitions = []
        for country in common_countries:
            start_rank = start_ranks[country]
            end_rank = end_ranks[country]
            rank_change = start_rank - end_rank  # Positive = improvement
            
            start_score = start_data[start_data['country'] == country]['cnps10_score'].iloc[0]
            end_score = end_data[end_data['country'] == country]['cnps10_score'].iloc[0]
            score_change = end_score - start_score
            
            transitions.append({
                'country': country,
                'start_rank': int(start_rank),
                'end_rank': int(end_rank),
                'rank_change': int(rank_change),
                'start_score': start_score,
                'end_score': end_score,
                'score_change': score_change,
                'change_percentage': (score_change / start_score) * 100 if start_score != 0 else 0
            })
        
        # Sort by rank improvement
        transitions.sort(key=lambda x: x['rank_change'], reverse=True)
        
        return {
            'transitions': transitions,
            'period': f"{start_year}-{end_year}",
            'total_countries': len(transitions)
        }
    
    def perform_correlation_analysis(self, year):
        """
        Perform correlation analysis between different power dimensions.
        
        Args:
            year (int): Year to analyze
            
        Returns:
            dict: Correlation analysis results
        """
        data = self.get_countries_by_year(year)
        
        if data.empty:
            return {}
        
        # Get dimension columns (excluding non-numeric columns)
        dimension_cols = ['economy', 'military', 'tech', 'diplomacy', 'governance', 'space', 'intelligence', 'scale', 'society']
        available_dims = [col for col in dimension_cols if col in data.columns]
        
        if len(available_dims) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_data = data[available_dims + ['cnps10_score']].corr()
        
        # Find strongest correlations with overall score
        score_correlations = corr_data['cnps10_score'].drop('cnps10_score').sort_values(ascending=False)
        
        return {
            'correlation_matrix': corr_data,
            'score_correlations': score_correlations,
            'dimensions': available_dims,
            'year': year
        }
    
    def generate_power_distribution_analysis(self, year):
        """
        Analyze power distribution across different tiers and regions.
        
        Args:
            year (int): Year to analyze
            
        Returns:
            dict: Power distribution analysis
        """
        data = self.get_countries_by_year(year)
        
        if data.empty:
            return {}
        
        # Tier analysis
        tier_analysis = {}
        if 'tier' in data.columns:
            tier_counts = data['tier'].value_counts()
            tier_avg_scores = data.groupby('tier')['cnps10_score'].mean().sort_values(ascending=False)
            tier_analysis = {
                'counts': tier_counts.to_dict(),
                'average_scores': tier_avg_scores.to_dict()
            }
        
        # Score distribution
        scores = data['cnps10_score']
        
        # Define power brackets
        def classify_power_level(score):
            if score >= 0.8: return "Superpower"
            elif score >= 0.6: return "Great Power"
            elif score >= 0.4: return "Major Power"
            elif score >= 0.2: return "Middle Power"
            else: return "Small Power"
        
        data['power_level'] = data['cnps10_score'].apply(classify_power_level)
        power_distribution = data['power_level'].value_counts()
        
        return {
            'tier_analysis': tier_analysis,
            'power_distribution': power_distribution.to_dict(),
            'total_countries': len(data),
            'year': year,
            'gini_coefficient': self._calculate_gini_coefficient(scores),
            'power_concentration_top10': scores.nlargest(10).sum() / scores.sum()
        }
    
    def _calculate_gini_coefficient(self, scores):
        """Calculate Gini coefficient for power concentration."""
        scores_sorted = np.sort(scores)
        n = len(scores_sorted)
        cumsum = np.cumsum(scores_sorted)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] != 0 else 0
    
    def create_dimension_radar_chart(self, countries, year):
        """
        Create radar chart comparing countries across different dimensions.
        
        Args:
            countries (list): List of country names to compare
            year (int): Year to analyze
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        data = self.get_countries_by_year(year)
        
        if data.empty:
            return go.Figure()
        
        # Filter data for selected countries
        country_data = data[data['country'].isin(countries)]
        
        # Dimension columns
        dimensions = ['economy', 'military', 'tech', 'diplomacy', 'governance', 'space', 'intelligence', 'scale', 'society']
        available_dims = [dim for dim in dimensions if dim in country_data.columns]
        
        if not available_dims:
            return go.Figure()
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, country in enumerate(countries):
            if country in country_data['country'].values:
                country_row = country_data[country_data['country'] == country].iloc[0]
                values = [country_row[dim] if dim in country_row else 0 for dim in available_dims]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_dims,
                    fill='toself',
                    name=country,
                    line_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Multi-Dimensional Power Comparison ({year})",
            title_font_size=18
        )
        
        return fig
    
    def calculate_global_statistics(self):
        """
        Calculate global statistics across all years.
        
        Returns:
            dict: Global statistics
        """
        if self.multi_year_data is None:
            # Use single year data if multi-year not available
            if self.latest_data is not None and 'cnps10_score' in self.latest_data.columns:
                scores = self.latest_data['cnps10_score']
                return {
                    'total_countries': len(scores),
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'score_range': scores.max() - scores.min(),
                    'percentile_10': scores.quantile(0.1),
                    'percentile_25': scores.quantile(0.25),
                    'percentile_50': scores.quantile(0.5),
                    'percentile_75': scores.quantile(0.75),
                    'percentile_90': scores.quantile(0.9)
                }
            return {}
        
        # Calculate across all years
        all_scores = self.multi_year_data['cnps10_score']
        
        return {
            'total_countries': len(self.multi_year_data['country'].unique()),
            'mean_score': all_scores.mean(),
            'std_score': all_scores.std(),
            'score_range': all_scores.max() - all_scores.min(),
            'percentile_10': all_scores.quantile(0.1),
            'percentile_25': all_scores.quantile(0.25),
            'percentile_50': all_scores.quantile(0.5),
            'percentile_75': all_scores.quantile(0.75),
            'percentile_90': all_scores.quantile(0.9)
        }

def main():
    """
    Main function to run the CNPS-10 web application.
    
    This function initializes the system, creates the user interface,
    and handles all user interactions.
    """
    
    # Display main header
    st.markdown('<div class="main-header">🌍 CNPS-10: National Power Assessment System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Open Source Research Platform for Power Analysis</div>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        system = CNPS10System()
    except Exception as e:
        st.error(f"❌ Failed to initialize CNPS-10 system: {str(e)}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.title("📊 Analysis Configuration")
    
    # Year selection
    available_years = system.get_available_years()
    
    st.sidebar.markdown("### 📅 Time Period Selection")
    
    # Single year analysis
    selected_year = st.sidebar.selectbox(
        "Select Year for Analysis",
        options=available_years,
        index=len(available_years)-1 if available_years else 0,
        help="Choose a specific year for detailed analysis"
    )
    
    # Multi-year trend analysis
    if len(available_years) > 1:
        st.sidebar.markdown("### 📈 Trend Analysis")
        
        year_range = st.sidebar.select_slider(
            "Select Year Range",
            options=available_years,
            value=(available_years[0], available_years[-1]),
            help="Choose start and end years for trend analysis"
        )
        
        start_year, end_year = year_range
    else:
        start_year, end_year = selected_year, selected_year
    
    # Country selection for trends
    if len(available_years) > 1:
        sample_countries = ['United States', 'China', 'Germany', 'United Kingdom', 'Japan', 'France', 'Russia', 'India']
        all_countries = []
        
        if system.multi_year_data is not None and 'country' in system.multi_year_data.columns:
            all_countries = sorted(system.multi_year_data['country'].unique())
        
        # Filter available countries
        available_sample = [c for c in sample_countries if c in all_countries]
        
        selected_countries = st.sidebar.multiselect(
            "Select Countries for Trend Analysis",
            options=all_countries,
            default=available_sample[:5],  # Default to first 5 available
            help="Choose countries to compare in trend analysis"
        )
    
    # Display options
    st.sidebar.markdown("### 🎨 Display Options")
    
    show_top_n = st.sidebar.slider(
        "Number of Top Countries to Display",
        min_value=5,
        max_value=30,
        value=15,
        help="Select how many top-ranked countries to show"
    )
    
    show_statistics = st.sidebar.checkbox(
        "Show Detailed Statistics",
        value=True,
        help="Display comprehensive statistical analysis"
    )
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Current Rankings", 
        "📈 Trend Analysis", 
        "🔬 Academic Analysis",
        "📋 Statistical Summary", 
        "🗺️ Geopolitical Mapping",
        "📄 Research Reports",
        "📖 Methodology",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header(f"🏆 CNPS-10 Rankings for {selected_year}")
        
        # Get data for selected year
        year_data = system.get_countries_by_year(selected_year)
        
        if not year_data.empty:
            # Display key metrics
            if 'cnps10_score' in year_data.columns:
                top_3 = system.get_top_countries(selected_year, 3)
                
                if len(top_3) >= 3:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card rank-1">
                                <h3>🥇 #{1}</h3>
                                <h2>{top_3.iloc[0]['country']}</h2>
                                <h1>{top_3.iloc[0]['cnps10_score']:.1%}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card rank-2">
                                <h3>🥈 #{2}</h3>
                                <h2>{top_3.iloc[1]['country']}</h2>
                                <h1>{top_3.iloc[1]['cnps10_score']:.1%}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card rank-3">
                                <h3>🥉 #{3}</h3>
                                <h2>{top_3.iloc[2]['country']}</h2>
                                <h1>{top_3.iloc[2]['cnps10_score']:.1%}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Ranking chart
            ranking_fig = system.create_ranking_chart(selected_year, show_top_n)
            if ranking_fig.data:
                st.plotly_chart(ranking_fig, width='stretch')
            
            # Detailed ranking table
            st.subheader(f"📋 Detailed Rankings ({selected_year})")
            
            top_countries = system.get_top_countries(selected_year, show_top_n)
            if not top_countries.empty:
                # Add ranking column
                top_countries_display = top_countries.copy()
                top_countries_display.reset_index(drop=True, inplace=True)
                top_countries_display.index += 1
                top_countries_display.index.name = 'Rank'
                
                # Select relevant columns for display
                display_columns = ['country', 'cnps10_score']
                if 'power_tier' in top_countries_display.columns:
                    display_columns.append('power_tier')
                
                # Format score as percentage
                top_countries_display['cnps10_score'] = top_countries_display['cnps10_score'].apply(lambda x: f'{x:.2%}')
                
                st.dataframe(
                    top_countries_display[display_columns],
                    column_config={
                        'country': 'Country',
                        'cnps10_score': 'CNPS-10 Score',
                        'power_tier': 'Power Tier'
                    },
                    width='stretch'
                )
        else:
            st.warning(f"⚠️ No data available for year {selected_year}")
    
    with tab2:
        if len(available_years) > 1:
            st.header(f"📈 Trend Analysis ({start_year}-{end_year})")
            
            if selected_countries:
                # Create trend chart
                trend_fig = system.create_trend_chart(selected_countries, start_year, end_year)
                if trend_fig.data:
                    st.plotly_chart(trend_fig, width='stretch')
                
                # Trend summary
                st.subheader("📊 Trend Summary")
                
                trend_data = []
                for country in selected_countries:
                    country_data = system.multi_year_data[
                        (system.multi_year_data['country'] == country) &
                        (system.multi_year_data['year'] >= start_year) &
                        (system.multi_year_data['year'] <= end_year)
                    ]
                    
                    if not country_data.empty and 'cnps10_score' in country_data.columns:
                        start_score = country_data[country_data['year'] == start_year]['cnps10_score'].iloc[0]
                        end_score = country_data[country_data['year'] == end_year]['cnps10_score'].iloc[0]
                        change = end_score - start_score
                        change_pct = change / start_score if start_score != 0 else 0
                        
                        trend_data.append({
                            'Country': country,
                            f'{start_year} Score': f'{start_score:.2%}',
                            f'{end_year} Score': f'{end_score:.2%}',
                            'Absolute Change': f'{change:.2%}',
                            'Relative Change': f'{change_pct:.1%}'
                        })
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    st.dataframe(trend_df, width='stretch')
            else:
                st.info("Please select countries in the sidebar for trend analysis.")
        else:
            st.info("Trend analysis requires multi-year data. Only single year data is available.")
    
    with tab3:
        st.header("📊 Academic Analysis")
        
        # Year selection for academic analysis
        col1, col2 = st.columns(2)
        with col1:
            analysis_year = st.selectbox(
                "Select Year for Analysis:",
                available_years,
                key="academic_year"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Correlation Analysis", "Power Transitions", "Distribution Analysis", "Radar Comparison"],
                key="analysis_type"
            )
        
        if analysis_type == "Correlation Analysis":
            st.subheader("Dimensional Correlation Analysis")
            
            corr_results = system.perform_correlation_analysis(analysis_year)
            if corr_results:
                # Correlation heatmap
                corr_matrix = corr_results['correlation_matrix']
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title=f"Dimension Correlation Matrix ({analysis_year})"
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, width="stretch")
                
                # Top correlations with overall score
                st.subheader("Dimensional Correlations with Overall CNPS Score")
                score_corr = corr_results['score_correlations']
                
                # Display all correlations in order
                st.write("**Correlation Strength Ranking:**")
                for i, (dim, corr) in enumerate(score_corr.items(), 1):
                    strength = "Very Strong" if abs(corr) >= 0.8 else "Strong" if abs(corr) >= 0.6 else "Moderate" if abs(corr) >= 0.4 else "Weak"
                    st.write(f"{i}. **{dim.title()}**: {corr:.3f} ({strength})")
                
                # Summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top 3 Strongest Correlations:**")
                    top_3 = score_corr.head(3)
                    for dim, corr in top_3.items():
                        st.write(f"• {dim.title()}: {corr:.3f}")
                
                with col2:
                    st.write("**Bottom 3 Correlations:**")
                    bottom_3 = score_corr.tail(3)
                    for dim, corr in bottom_3.items():
                        st.write(f"• {dim.title()}: {corr:.3f}")
                
                # Additional insights
                avg_correlation = score_corr.mean()
                st.info(f"**Average correlation with overall score: {avg_correlation:.3f}**")
                
                if avg_correlation > 0.7:
                    st.success("High dimensional coherence - all dimensions strongly contribute to overall power.")
                elif avg_correlation > 0.5:
                    st.warning("Moderate dimensional coherence - most dimensions contribute to overall power.")
                else:
                    st.error("Low dimensional coherence - dimensions may be measuring different aspects of power.")
        
        elif analysis_type == "Power Transitions":
            st.subheader("Power Transition Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.selectbox("Start Year:", available_years, key="trans_start")
            with col2:
                end_years = [y for y in available_years if y > start_year]
                if end_years:
                    end_year = st.selectbox("End Year:", end_years, key="trans_end")
                else:
                    st.warning("Please select a start year with available subsequent years.")
                    end_year = None
            
            if end_year:
                transitions = system.calculate_power_transitions(start_year, end_year)
                if transitions and transitions['transitions']:
                    st.write(f"**Analysis Period: {transitions['period']}**")
                    st.write(f"Countries Analyzed: {transitions['total_countries']}")
                    
                    # Top gainers and losers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🚀 Top Power Gainers (Rank Improvement):**")
                        gainers = [t for t in transitions['transitions'] if t['rank_change'] > 0][:10]
                        for t in gainers:
                            st.write(f"• {t['country']}: +{t['rank_change']} ranks ({t['change_percentage']:+.1f}%)")
                    
                    with col2:
                        st.write("**📉 Top Power Decliners (Rank Decline):**")
                        decliners = [t for t in transitions['transitions'] if t['rank_change'] < 0][-10:]
                        for t in decliners:
                            st.write(f"• {t['country']}: {t['rank_change']} ranks ({t['change_percentage']:+.1f}%)")
        
        elif analysis_type == "Distribution Analysis":
            st.subheader("Power Distribution Analysis")
            
            dist_analysis = system.generate_power_distribution_analysis(analysis_year)
            if dist_analysis:
                # Power level distribution
                power_dist = dist_analysis['power_distribution']
                
                fig_dist = px.pie(
                    values=list(power_dist.values()),
                    names=list(power_dist.keys()),
                    title=f"Global Power Distribution ({analysis_year})"
                )
                st.plotly_chart(fig_dist, width="stretch")
                
                # Power concentration metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gini Coefficient", f"{dist_analysis['gini_coefficient']:.3f}")
                with col2:
                    st.metric("Top 10 Power Share", f"{dist_analysis['power_concentration_top10']:.1%}")
                with col3:
                    st.metric("Total Countries", dist_analysis['total_countries'])
        
        elif analysis_type == "Radar Comparison":
            st.subheader("Multi-Dimensional Country Comparison")
            
            # Country selection for radar chart
            available_countries = system.get_countries_by_year(analysis_year)['country'].tolist()
            selected_countries = st.multiselect(
                "Select Countries to Compare (max 5):",
                available_countries,
                default=available_countries[:3] if len(available_countries) >= 3 else available_countries,
                max_selections=5
            )
            
            if selected_countries:
                radar_chart = system.create_dimension_radar_chart(selected_countries, analysis_year)
                st.plotly_chart(radar_chart, width="stretch")
        
        else:
            st.info("Please select an analysis type to begin academic analysis.")
    
    with tab4:
        st.header("📈 Statistical Summary")
        
        if show_statistics:
            # Global statistics
            global_stats = system.calculate_global_statistics()
            if global_stats:
                st.subheader("Global Power Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Countries", global_stats['total_countries'])
                with col2:
                    st.metric("Average Score", f"{global_stats['mean_score']:.3f}")
                with col3:
                    st.metric("Standard Deviation", f"{global_stats['std_score']:.3f}")
                with col4:
                    st.metric("Score Range", f"{global_stats['score_range']:.3f}")
                
                # Percentile analysis
                st.subheader("Score Percentiles")
                percentiles_df = pd.DataFrame({
                    'Percentile': ['10th', '25th', '50th', '75th', '90th'],
                    'Score': [
                        global_stats['percentile_10'],
                        global_stats['percentile_25'],
                        global_stats['percentile_50'],
                        global_stats['percentile_75'],
                        global_stats['percentile_90']
                    ]
                })
                st.dataframe(percentiles_df, width="stretch")
            
            # Single year detailed statistics
            stats = system.generate_statistics_summary(selected_year)
            if stats:
                st.subheader(f"Detailed Statistics for {selected_year}")
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Score", f"{stats['mean_score']:.2%}")
                    st.metric("Median Score", f"{stats['median_score']:.2%}")
                
                with col2:
                    st.metric("Standard Deviation", f"{stats['std_score']:.2%}")
                    st.metric("Coefficient of Variation", f"{stats['std_score']/stats['mean_score']:.3f}")
                
                with col3:
                    st.metric("Minimum Score", f"{stats['min_score']:.2%}")
                    st.metric("25th Percentile", f"{stats['q1_score']:.2%}")
                
                with col4:
                    st.metric("Maximum Score", f"{stats['max_score']:.2%}")
                    st.metric("75th Percentile", f"{stats['q3_score']:.2%}")
                
                # Advanced statistics
                st.subheader("📊 Advanced Statistical Measures")
                
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    st.metric(
                        "Skewness", 
                        f"{stats['skewness']:.3f}",
                        help="Measure of asymmetry in the score distribution"
                    )
                
                with adv_col2:
                    st.metric(
                        "Kurtosis", 
                        f"{stats['kurtosis']:.3f}",
                        help="Measure of tail heaviness in the score distribution"
                    )
                
                # Distribution analysis
                year_data = system.get_countries_by_year(selected_year)
                if not year_data.empty and 'cnps10_score' in year_data.columns:
                    st.subheader("📈 Score Distribution")
                    
                    hist_fig = px.histogram(
                        year_data,
                        x='cnps10_score',
                        nbins=20,
                        title=f'Distribution of CNPS-10 Scores ({selected_year})',
                        labels={'cnps10_score': 'CNPS-10 Score (%)', 'count': 'Number of Countries'}
                    )
                    
                    hist_fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        height=400
                    )
                    
                    hist_fig.update_xaxes(tickformat='.1%')
                    st.plotly_chart(hist_fig, width="stretch")
            
            # Yearly comparison
            if system.multi_year_data is not None:
                st.subheader("Year-over-Year Analysis")
                yearly_stats = []
                
                for year in available_years:
                    year_data = system.get_countries_by_year(year)
                    if not year_data.empty:
                        yearly_stats.append({
                            'Year': year,
                            'Countries': len(year_data),
                            'Mean Score': year_data['cnps10_score'].mean(),
                            'Max Score': year_data['cnps10_score'].max(),
                            'Min Score': year_data['cnps10_score'].min(),
                            'Std Dev': year_data['cnps10_score'].std()
                        })
                
                if yearly_stats:
                    yearly_df = pd.DataFrame(yearly_stats)
                    st.dataframe(yearly_df, width="stretch")
                    
                    # Trend chart
                    fig_trend = px.line(yearly_df, x='Year', y='Mean Score', 
                                      title="Global Average CNPS Score Trend")
                    st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info("Enable 'Show Detailed Statistics' in the sidebar to view statistical analysis.")
    with tab5:
        st.header("🌍 Geopolitical Mapping")
        
        # Year selection
        map_year = st.selectbox("Select Year for Mapping:", available_years, key="map_year")
        
        # Get data for selected year
        map_data = system.get_countries_by_year(map_year)
        
        if not map_data.empty:
            # Regional analysis
            st.subheader("Regional Power Distribution")
            
            # Define regions (simplified)
            region_mapping = {
                'United States': 'North America',
                'China': 'Asia-Pacific',
                'Japan': 'Asia-Pacific',
                'Germany': 'Europe',
                'United Kingdom': 'Europe',
                'France': 'Europe',
                'India': 'Asia-Pacific',
                'Italy': 'Europe',
                'Brazil': 'South America',
                'Canada': 'North America',
                'Russia': 'Europe',
                'South Korea': 'Asia-Pacific',
                'Australia': 'Asia-Pacific',
                'Spain': 'Europe',
                'Mexico': 'North America',
                'Indonesia': 'Asia-Pacific',
                'Netherlands': 'Europe',
                'Saudi Arabia': 'Middle East',
                'Turkey': 'Middle East',

                'Switzerland': 'Europe',
                'Belgium': 'Europe',
                'Israel': 'Middle East',
                'Poland': 'Europe',
                'Ireland': 'Europe',
                'Austria': 'Europe',
                'Argentina': 'South America',
                'Thailand': 'Asia-Pacific',
                'Nigeria': 'Africa',
                'Egypt': 'Africa',
                'South Africa': 'Africa',
                'Philippines': 'Asia-Pacific',
                'Bangladesh': 'Asia-Pacific',
                'Chile': 'South America',
                'Finland': 'Europe',
                'Romania': 'Europe',
                'Czech Republic': 'Europe',
                'Portugal': 'Europe',
                'Peru': 'South America',
                'New Zealand': 'Asia-Pacific',
                'Greece': 'Europe',
                'Iraq': 'Middle East',
                'Algeria': 'Africa',
                'Kazakhstan': 'Asia-Pacific',
                'Qatar': 'Middle East',
                'Hungary': 'Europe',
                'Kuwait': 'Middle East',
                'Morocco': 'Africa',
                'Slovakia': 'Europe',
                'Ecuador': 'South America',
                'Dominican Republic': 'North America',
                'Ethiopia': 'Africa',
                'Guatemala': 'North America',
                'Uzbekistan': 'Asia-Pacific',
                'Myanmar': 'Asia-Pacific',
                'Sri Lanka': 'Asia-Pacific',
                'Kenya': 'Africa',
                'Uruguay': 'South America',
                'Costa Rica': 'North America',
                'Slovenia': 'Europe',
                'Lithuania': 'Europe',
                'Tunisia': 'Africa',
                'Panama': 'North America',
                'Serbia': 'Europe',
                'Jordan': 'Middle East',
                'Ghana': 'Africa',
                'Ivory Coast': 'Africa',
                'Tanzania': 'Africa',
                'Cameroon': 'Africa',
                'Bolivia': 'South America',
                'Uganda': 'Africa',
                'Latvia': 'Europe',
                'Estonia': 'Europe',
                'Paraguay': 'South America',
                'Zambia': 'Africa',
                'Cambodia': 'Asia-Pacific',
                'Botswana': 'Africa',
                'Senegal': 'Africa',
                'Honduras': 'North America',
                'Trinidad and Tobago': 'North America',
                'Nicaragua': 'North America',
                'Moldova': 'Europe',
                'Madagascar': 'Africa',
                'Malta': 'Europe',
                'Bahrain': 'Middle East',
                'Jamaica': 'North America',
                'Mongolia': 'Asia-Pacific',
                'Armenia': 'Asia-Pacific',
                'Namibia': 'Africa',
                'Mauritius': 'Africa',
                'North Macedonia': 'Europe',
                'Albania': 'Europe',
                'Brunei': 'Asia-Pacific',
                'Georgia': 'Asia-Pacific',
                'Gabon': 'Africa',
                'Iceland': 'Europe',
                'Barbados': 'North America',
                'Montenegro': 'Europe',
                'Suriname': 'South America',
                'Mauritania': 'Africa',
                'Cyprus': 'Europe',
                'Eswatini': 'Africa',
                'Fiji': 'Asia-Pacific',
                'Maldives': 'Asia-Pacific',
                'Guyana': 'South America',
                'Bahamas': 'North America',
                'Belize': 'North America',
                'Djibouti': 'Africa',
                'Timor-Leste': 'Asia-Pacific',
                'Comoros': 'Africa',
                'Solomon Islands': 'Asia-Pacific',
                'Seychelles': 'Africa',
                'Luxembourg': 'Europe',
                'Vanuatu': 'Asia-Pacific',
                'Cape Verde': 'Africa',
                'Samoa': 'Asia-Pacific',
                'Saint Lucia': 'North America',
                'Kiribati': 'Asia-Pacific',
                'Micronesia': 'Asia-Pacific',
                'Grenada': 'North America',
                'Saint Vincent and the Grenadines': 'North America',
                'Tonga': 'Asia-Pacific',
                'Saint Kitts and Nevis': 'North America',
                'Dominica': 'North America',
                'Palau': 'Asia-Pacific',
                'Marshall Islands': 'Asia-Pacific',
                'San Marino': 'Europe',
                'Liechtenstein': 'Europe',
                'Monaco': 'Europe',
                'Nauru': 'Asia-Pacific',
                'Andorra': 'Europe',
                'Tuvalu': 'Asia-Pacific',
                'Vatican City': 'Europe'
            }
            
            # Add region column
            map_data['region'] = map_data['country'].map(region_mapping)
            map_data['region'] = map_data['region'].fillna('Other')
            
            # Regional statistics
            regional_stats = map_data.groupby('region').agg({
                'cnps10_score': ['count', 'mean', 'max', 'min', 'sum']
            }).round(3)
            regional_stats.columns = ['Countries', 'Avg Score', 'Max Score', 'Min Score', 'Total Power']
            
            st.dataframe(regional_stats, width="stretch")
            
            # Regional comparison chart
            fig_regional = px.box(map_data, x='region', y='cnps10_score',
                                title=f"Regional Power Distribution ({map_year})")
            fig_regional.update_xaxes(tickangle=45)
            st.plotly_chart(fig_regional, width="stretch")
            
            # Power balance analysis
            st.subheader("Global Power Balance")
            
            # Top powers by tier
            if 'tier' in map_data.columns:
                tier_analysis = map_data.groupby('tier').agg({
                    'country': 'count',
                    'cnps10_score': 'mean'
                }).round(3)
                tier_analysis.columns = ['Number of Countries', 'Average Score']
                st.dataframe(tier_analysis, width="stretch")
            
            # Power concentration
            st.subheader("Power Concentration Analysis")
            top_10_share = map_data.nlargest(10, 'cnps10_score')['cnps10_score'].sum() / map_data['cnps10_score'].sum()
            top_5_share = map_data.nlargest(5, 'cnps10_score')['cnps10_score'].sum() / map_data['cnps10_score'].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top 10 Countries Power Share", f"{top_10_share:.1%}")
            with col2:
                st.metric("Top 5 Countries Power Share", f"{top_5_share:.1%}")
        else:
            st.info("No data available for the selected year.")
    
    with tab6:
        st.header("📄 Research Reports")
        
        st.subheader("Academic Research Framework")
        
        # Research methodology
        with st.expander("🔬 Methodology & Framework"):
            st.markdown("""
            ### CNPS-10 Research Methodology
            
            **Theoretical Foundation:**
            - Multi-dimensional power assessment framework
            - Quantitative analysis of national capabilities
            - Temporal trend analysis and forecasting
            
            **Key Dimensions Analyzed:**
            1. **Economic Power**: GDP, trade, financial capabilities
            2. **Military Power**: Defense spending, capabilities, technology
            3. **Technological Power**: Innovation, R&D, digital infrastructure
            4. **Diplomatic Power**: International relations, soft power
            5. **Governance**: Institutional quality, effectiveness
            6. **Space Capabilities**: Space technology and assets
            7. **Intelligence**: Information capabilities and cyber power
            8. **Scale**: Geographic and demographic factors
            9. **Social Power**: Cultural influence, education, social cohesion
            
            **Statistical Methods:**
            - Correlation analysis between dimensions
            - Power transition modeling
            - Distribution analysis using Gini coefficients
            - Longitudinal trend analysis
            """)
        
        # Generate research summary
        with st.expander("📊 Current Research Findings"):
            report_year = st.selectbox("Generate Report for Year:", available_years, key="report_year")
            
            if st.button("Generate Research Summary"):
                # Generate comprehensive analysis
                year_data = system.get_countries_by_year(report_year)
                
                if not year_data.empty:
                    st.markdown(f"""
                    ### Research Summary for {report_year}
                    
                    **Sample Size:** {len(year_data)} countries analyzed
                    
                    **Key Findings:**
                    """)
                    
                    # Top performers
                    top_5 = year_data.nlargest(5, 'cnps10_score')
                    st.write("**Top 5 Powers:**")
                    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                        st.write(f"{i}. {row['country']}: {row['cnps10_score']:.3f}")
                    
                    # Statistical analysis
                    corr_results = system.perform_correlation_analysis(report_year)
                    if corr_results:
                        st.write("**Strongest Dimension Correlations with Overall Power:**")
                        top_corr = corr_results['score_correlations'].head(3)
                        for dim, corr in top_corr.items():
                            st.write(f"• {dim.title()}: {corr:.3f}")
                    
                    # Power distribution
                    dist_analysis = system.generate_power_distribution_analysis(report_year)
                    if dist_analysis:
                        st.write(f"**Power Concentration Metrics:**")
                        st.write(f"• Gini Coefficient: {dist_analysis['gini_coefficient']:.3f}")
                        st.write(f"• Top 10 Power Share: {dist_analysis['power_concentration_top10']:.1%}")
        
        # Data export options
        with st.expander("💾 Data Export Options"):
            st.markdown("""
            ### Export Options for Academic Research
            
            **Available Data Formats:**
            - CSV: Raw data for statistical analysis
            - JSON: Structured data for programming
            - Academic citations and references
            """)
            
            export_year = st.selectbox("Select Year for Export:", available_years, key="export_year")
            
            if st.button("Prepare Data Export"):
                export_data = system.get_countries_by_year(export_year)
                if not export_data.empty:
                    st.success(f"Data prepared for {export_year}: {len(export_data)} countries")
                    st.write("Data preview:")
                    st.dataframe(export_data.head(), width="stretch")
        
        # Academic citations
        with st.expander("📚 Academic References & Citations"):
            st.markdown("""
            ### Suggested Citation Format
            
            **APA Style:**
            CNPS Research Team. (2025). Comprehensive National Power Score (CNPS-10) Database. 
            Academic Research Platform. Retrieved from [URL]
            
            **Chicago Style:**
            CNPS Research Team. "Comprehensive National Power Score (CNPS-10) Database." 
            Academic Research Platform, 2025. [URL]
            
            ### Related Academic Literature
            
            **Core Concepts:**
            - National power measurement theory
            - Quantitative international relations
            - Power transition theory
            - Capability aggregation methodologies
            
            **Methodological References:**
            - Multi-dimensional scaling in political science
            - Comparative politics quantitative methods
            - International relations statistical analysis
            """)
    
    with tab7:
        st.header("📖 Methodology")
        
        st.markdown("""
        ### Data Collection and Processing Methodology
        
        #### Data Sources and Transparency
        
        **Data Generation Approach:**  
        The CNPS-10 system uses a realistic modeling approach that combines:
        - Publicly available national statistics
        - Historical patterns and trends
        - Statistical modeling for predictive analysis
        
        **Note on Data Nature:**  
        This is an **open-source research project** using modeled data for academic analysis and demonstration purposes. 
        The rankings reflect commonly understood global power dynamics but should not be considered official or 
        authoritative assessments.
        
        #### Nine Dimensions of National Power
        
        1. **Economic Power** - GDP, trade volumes, economic stability
        2. **Military Strength** - Defense capabilities, military expenditure
        3. **Technological Innovation** - R&D investment, patents, high-tech exports
        4. **Diplomatic Influence** - International relationships, soft power
        5. **Governance Quality** - Institutional effectiveness, rule of law
        6. **Space Capabilities** - Space programs, satellite infrastructure
        7. **Intelligence Systems** - Information gathering and analysis capabilities
        8. **Geographic Scale** - Territory, population, natural resources
        9. **Social Power** - Cultural influence, education, social cohesion
        
        #### Mathematical Framework
        
        **CNPS-10 Score Calculation:**
        ```
        CNPS-10 = Σ(Dimension_i × Weight_i × Tier_Multiplier)
        ```
        
        **Power Tier Classifications:**
        - **Superpower** (≥0.75): Global influence across all dimensions
        - **Great Power** (0.65-0.74): Regional dominance, global influence in some areas
        - **Major Power** (0.55-0.64): Significant regional influence
        - **Regional Power** (0.45-0.54): Important within region
        - **Middle Power** (0.35-0.44): Moderate influence
        - **Small Power** (<0.35): Limited influence
        
        #### Limitations and Considerations
        
        ⚠️ **Important Disclaimers:**
        - This is a **research demonstration project**, not an official assessment
        - Data is modeled for educational and research purposes
        - Rankings are approximate and should not be used for policy decisions
        - Actual national power involves qualitative factors not captured in quantitative models
        
        #### Technical Implementation
        
        **Data Processing:**
        - High-precision calculations (float64)
        - Temporal analysis (2000-2050)
        - Statistical validation and testing
        - Memory optimization for large datasets
        
        **Visualization:**
        - Interactive charts using Plotly
        - Streamlit web framework
        - Real-time analysis capabilities
        """)
        
        st.info("""
        **Academic Usage:** This tool is designed for educational purposes, research demonstrations, 
        and as a starting point for more sophisticated national power assessment projects.
        """)
    
    with tab8:
        
        st.header("ℹ️ About CNPS-10 System")
        
        st.markdown("""
        ### CNPS-10: National Power Assessment System
        
        **Version:** 1.0 Open Source Research Platform  
        **Last Updated:** September 2025  
        **Data Coverage:** 174 countries worldwide  
        **Time Range:** 2000-2050 (modeled data)
        
        #### Project Overview
        The CNPS-10 system is an **open-source research project** that demonstrates multi-dimensional 
        analysis of national capabilities. This platform serves as an educational tool and research 
        framework for understanding how different factors might contribute to national power assessment.
        
        **⚠️ Important:** This is a research demonstration using modeled data for educational purposes. 
        Rankings should not be considered authoritative assessments.
        
        #### Technical Implementation
        
        **Built with:**
        - Python for data processing and analysis
        - Streamlit for web interface
        - Plotly for interactive visualizations
        - Pandas & NumPy for data manipulation
        
        #### Open Source Information
        
        - **GitHub Repository:** https://github.com/Kelvin927/CNPS-10
        - **License:** MIT Open Source
        - **Purpose:** Educational and research demonstration
        
        #### Educational Goals
        
        This tool demonstrates:
        - Multi-dimensional data analysis techniques
        - Interactive web application development
        - Statistical visualization methods
        - Research methodology concepts
        """)
        
        # Technical information
        st.subheader("🔧 Technical Information")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.info(f"""
            **System Version**: 1.0.0  
            **Data Version**: {datetime.now().strftime('%Y-%m-%d')}  
            **Countries Covered**: {len(system.get_countries_by_year(selected_year))}  
            **Time Range**: {min(available_years)}-{max(available_years)}
            """)
        
        with tech_col2:
            st.info(f"""
            **Framework**: Streamlit {st.__version__}  
            **Data Processing**: pandas, numpy  
            **Visualization**: plotly  
            **License**: MIT Open Source
            """)

if __name__ == "__main__":
    main()
