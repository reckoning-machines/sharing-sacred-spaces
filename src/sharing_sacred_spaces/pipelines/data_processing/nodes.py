import pandas as pd
import plotly.express as px


def preprocess_fbi_data(fbi_data: pd.DataFrame) -> pd.DataFrame:
    """Filters hate crimes for specific religious biases and location types.

    Args:
        fbi_data: Raw FBI data.

    Returns:
        Filtered religious hate crime data.
    """
    religious_biases = [
        "Anti-Jewish",
        "Anti-Protestant",
        "Anti-Catholic",
        "Anti-Islamic",
        "Anti-Other Christian",
        "Anti-Hindu",
        "Anti-Sikh",
        "Anti-Buddhist",
        "Anti-Church",
    ]
    preprocessed_fbi_data = fbi_data[
        fbi_data["bias_desc"].str.contains("|".join(religious_biases), case=False)
        | (fbi_data["location_name"] == "Church/Synagogue/Temple/Mosque")
    ]
    return preprocessed_fbi_data


def incidents_by_city_total(hate_crime_data: pd.DataFrame):
    """Generates a bar plot for the number of hate crime incidents by city.

    Args:
        hate_crime_data: Filtered religious hate crime data.
    """
    # Filter to include only cities
    city_data = hate_crime_data[hate_crime_data["agency_type_name"] == "City"]

    city_incidents = (
        city_data.groupby("pug_agency_name").size().reset_index(name="incident_count")
    )
    fig = px.bar(
        city_incidents,
        x="pug_agency_name",
        y="incident_count",
        title="Number of Religious Hate Crime Incidents by City",
    )
    fig.show()


def incidents_by_state_total(hate_crime_data: pd.DataFrame):
    """Generates a bar plot for the number of hate crime incidents by state.

    Args:
        hate_crime_data: Filtered religious hate crime data.
    """
    state_incidents = (
        hate_crime_data.groupby("state_name").size().reset_index(name="incident_count")
    )
    fig = px.bar(
        state_incidents,
        x="state_name",
        y="incident_count",
        title="Number of Religious Hate Crime Incidents by State",
    )
    fig.show()


def top_5_cities_trends(hate_crime_data: pd.DataFrame):
    """Generates a line plot for the top 5 cities' hate crime trends over time.

    Args:
        hate_crime_data: Filtered religious hate crime data.
    """
    top_cities = hate_crime_data["pug_agency_name"].value_counts().head(5).index
    top_cities_data = hate_crime_data[
        hate_crime_data["pug_agency_name"].isin(top_cities)
    ]
    trends_by_city = (
        top_cities_data.groupby(["data_year", "pug_agency_name"])
        .size()
        .reset_index(name="incident_count")
    )
    fig = px.line(
        trends_by_city,
        x="data_year",
        y="incident_count",
        color="pug_agency_name",
        title="Top 5 Cities Religious Hate Crime Trends Over Time",
    )
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
