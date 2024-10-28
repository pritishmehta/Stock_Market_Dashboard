import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

def fetch_india_gdp_data():
    """
    Fetches India's GDP growth data from World Bank API
    Indicator: NY.GDP.MKTP.KD.ZG (GDP growth annual %)
    """
    # World Bank API endpoint
    base_url = "http://api.worldbank.org/v2/country/IND/indicator/NY.GDP.MKTP.KD.ZG"
    params = {
        "format": "json",
        "per_page": 100,  # Get more years of data
        "date": "2000:2024"  # Data range
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()[1]  # Second element contains the actual data
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'])
        df['date'] = pd.to_numeric(df['date'])
        df = df[['date', 'value']].sort_values('date')
        df.columns = ['Year', 'GDP Growth (%)']
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def create_gdp_visualization(df):
    """
    Creates an interactive visualization using plotly
    """
    if df is None or df.empty:
        return None
    
    fig = px.line(df, 
                  x='Year', 
                  y='GDP Growth (%)',
                  title='India GDP Growth Rate (Annual %)',
                  template='plotly_white')
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="GDP Growth Rate (%)",
        hovermode='x',
        showlegend=False
    )
    
    # Add markers to the line
    fig.update_traces(mode='lines+markers')
    
    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return st.plotly(fig)

def analyze_gdp_trends(df):
    """
    Provides statistical analysis of GDP trends
    """
    if df is None or df.empty:
        return None
    
    analysis = {
        'Latest Growth Rate': f"{df['GDP Growth (%)'].iloc[-1]:.1f}%",
        'Average Growth (Last 5 Years)': f"{df['GDP Growth (%)'].tail(5).mean():.1f}%",
        'Highest Growth': f"{df['GDP Growth (%)'].max():.1f}% ({df.loc[df['GDP Growth (%)'].idxmax(), 'Year']:.0f})",
        'Lowest Growth': f"{df['GDP Growth (%)'].min():.1f}% ({df.loc[df['GDP Growth (%)'].idxmin(), 'Year']:.0f})",
        'Standard Deviation': f"{df['GDP Growth (%)'].std():.1f}%"
    }
    
    return analysis

# Main execution
if __name__ == "__main__":
    # Fetch data
    df = fetch_india_gdp_data()
    
    if df is not None:
        # Create visualization
        fig = create_gdp_visualization(df)
        if fig:
            # Save to HTML file
            fig.write_html("india_gdp_growth.html")
            
            # Optionally display in browser
            fig.show()
        
        # Print analysis
        analysis = analyze_gdp_trends(df)
        if analysis:
            print("\nGDP Growth Analysis:")
            for key, value in analysis.items():
                print(f"{key}: {value}")
    else:
        print("Failed to fetch GDP data")