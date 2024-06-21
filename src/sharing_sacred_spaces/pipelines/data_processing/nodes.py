import pandas as pd
import plotly.express as px

def preprocess_fbi_data(fbi_data: pd.DataFrame) -> pd.DataFrame:
    # Implement your hate crime filter logic here
    # For example, you could filter by specific offense types related to hate crimes
    hate_crime_data = fbi_data[fbi_data['offense_name'].str.contains('Hate Crime', case=False)]
    return hate_crime_data

def incidents_by_city_total(hate_crime_data: pd.DataFrame):
    city_incidents = hate_crime_data.groupby('location_name').size().reset_index(name='incident_count')
    fig = px.bar(city_incidents, x='location_name', y='incident_count', title='Number of Hate Crime Incidents by City')
    fig.show()

def incidents_by_state_total(hate_crime_data: pd.DataFrame):
    state_incidents = hate_crime_data.groupby('state_abbr').size().reset_index(name='incident_count')
    fig = px.bar(state_incidents, x='state_abbr', y='incident_count', title='Number of Hate Crime Incidents by State')
    fig.show()

def top_5_cities_trends(hate_crime_data: pd.DataFrame):
    top_cities = hate_crime_data['location_name'].value_counts().head(5).index
    top_cities_data = hate_crime_data[hate_crime_data['location_name'].isin(top_cities)]
    fig = px.line(top_cities_data, x='data_year', y='incident_id', color='location_name',
                  title='Top 5 Cities Hate Crime Trends Over Time')
    fig.show()

def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
