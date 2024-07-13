import pandas as pd
import plotly.express as px

pd.set_option("mode.chained_assignment", None)


def create_dataset(
    preprocessed_fbi_data: pd.DataFrame, preprocessed_gini_data: pd.DataFrame
) -> pd.DataFrame:
    df_fbi_merge = preprocessed_fbi_data[["year", "counter", "city", "state"]]
    df_fbi_merge["state"] = df_fbi_merge["state"].str.strip()
    preprocessed_gini_data["state"] = preprocessed_gini_data["state"].str.strip()
    dataset = df_fbi_merge.merge(
        preprocessed_gini_data,
        how="inner",
        left_on=["year", "city", "state"],
        right_on=["year", "city", "state"],
    )
    return dataset


def preprocess_gini_data():
    import os

    root = os.getcwd()
    root = root.split("/")[:-1]
    root = "/".join(root[1:])
    root = "/" + root
    root = root + "/sharing-sacred-spaces/data/01_raw/"
    import glob

    file_list = glob.glob(root + "*-Data.csv")
    df_list = []
    for file_name in file_list:
        # print(file_name)
        year_name = file_name.split("/")[8]
        year_name = year_name.replace("ACSDT1Y", "")
        year_name = year_name.replace(".B19083-Data.csv", "")
        df_gini = pd.read_csv(file_name)
        df_gini.dropna(axis=1, how="all", inplace=True)

        df_gini.columns = df_gini.iloc[0]
        df_gini.drop(df_gini.index[0], inplace=True)
        df_gini["year"] = year_name
        df_list.append(df_gini)
    df = pd.concat(df_list)
    df[["city", "state"]] = df["Geographic Area Name"].str.split(",", expand=True)
    df["city"] = df["city"].str.replace(" city", "")
    df["city"] = df["city"].str.replace(" zona urbana", "")
    keep_cols = [
        "Estimate!!Gini Index",
        "Margin of Error!!Gini Index",
        "year",
        "city",
        "state",
    ]
    df = df[keep_cols]
    df.columns = ["gini_index", "margin_of_error_gini_index", "year", "city", "state"]
    return df


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
        & (fbi_data["location_name"] == "Church/Synagogue/Temple/Mosque")
    ]
    preprocessed_fbi_data = preprocessed_fbi_data.loc[
        preprocessed_fbi_data["agency_type_name"] != "County"
    ]
    preprocessed_fbi_data["year"] = preprocessed_fbi_data["data_year"]
    keep_cols = [
        "year",
        # "pug_agency_name",
        "agency_type_name",
        # "state_name",
        "population_group_description",
        "incident_date",
        "offense_name",
        "location_name",
        "bias_desc",
        "victim_types",
        "counter",
        "city",
        "state",
    ]
    preprocessed_fbi_data["counter"] = 1
    preprocessed_fbi_data["city"] = preprocessed_fbi_data["pug_agency_name"]
    preprocessed_fbi_data["state"] = preprocessed_fbi_data["state_name"]

    return preprocessed_fbi_data[keep_cols]


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
