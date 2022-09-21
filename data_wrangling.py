"""
This file contains the functions for data wrangling
"""

import pandas

province_name = {"Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador",
                 "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"}


def clean_food_price_data(name: str, dataset: pandas.Series, filters: str, scale: float) -> pandas.Series:
    """Return properly formatted dataframe after data wrangling for the food price datasets

    name: name of the dataframe
    dataset: raw dataframe
    filter: specific category
    scale: resize the value, e.g. 0.001 will scale down values from millions to thousands
    """
    columns = dataset.keys().tolist()
    dataset.loc[dataset["VALUE"].isnull(), "is_NaN"] = "YES"
    dataset.loc[dataset["VALUE"].notnull(), "is_NaN"] = "NO"

    # Filter
    # for i, row in dataset.iterrows():
    #     if dataset.at[i, "GEO"] == "Yukon":
    #         dataset.at[i, "GEO"] = "Yukon Territory"
    dataset = dataset[dataset["GEO"].isin(province_name)]
    dataset = dataset[dataset[columns[3]] == filters]

    # NA Values
    for i, row in dataset.iterrows():
        if dataset.at[i, "is_NaN"] == "YES":
            specific_geo = dataset[dataset["GEO"] == dataset.at[i, "GEO"]]
            specific_geo = specific_geo[specific_geo.VALUE.notnull()]
            average = sum(specific_geo["VALUE"]) / len(specific_geo)
            dataset.at[i, "VALUE"] = average
        dataset.at[i, "VALUE"] = dataset.at[i, "VALUE"] * scale

    # Select column & Sort
    column = ["REF_DATE", "GEO", "VALUE"]
    dataset = dataset[column]
    dataset.sort_values(["REF_DATE", "GEO"], ascending=(True, True))

    # Output
    dataset.to_csv(name + ".csv")
    return dataset


def clean_case_data(name: str, dataset: pandas.Series) -> pandas.Series:
    """Return properly formatted dataframe after data wrangling for the covid datasets

    name: name of the dataset
    dataset: raw dataframe
    """
    # Select Column and Sort
    column = ["province", "date_report", "cases"]
    dataset = dataset[column]
    dataset.columns = ["GEO", "date_report", "VALUE"]
    dataset.sort_values(["GEO", "date_report"], ascending=(True, True))
    match_province = {"BC": "British Columbia", "NL": "Newfoundland and Labrador", "NWT": "Northwest Territories",
                      "PEI": "Prince Edward Island"}

    # Filter
    for i, row in dataset.iterrows():
        if not dataset.at[i, "date_report"].startswith("25"):
            dataset = dataset.drop(i)
        elif dataset.at[i, "date_report"] == "25-10-2021" or dataset.at[i, "date_report"] == "25-11-2021":
            dataset = dataset.drop(i)
        elif dataset.at[i, "GEO"] in match_province:
            dataset.at[i, "GEO"] = match_province[dataset.at[i, "GEO"]]
        elif dataset.at[i, "GEO"] == 'Repatriated':
            dataset = dataset.drop(i)
    dataset = dataset[dataset["GEO"].isin(province_name)]

    # Output
    dataset.to_csv(name + ".csv")
    return dataset


def add_id_column(dataset: pandas.DataFrame, province_to_id: dict) -> pandas.DataFrame:
    """Return a dataset with a province id column appended to it.

        Preconditions:
          - len(dataset) != 0
    """
    # Adding an id column to our cases data
    ids = []
    for province in dataset['GEO']:
        ids.append(province_to_id[province])

    # Adding the new columns to data frame
    dataset['id'] = ids

    return dataset


def to_ref_date(delta: int) -> str:
    """Return the string version of the reference date based on the number of months passed after 2020-01
    """
    month = delta % 12 + 1
    year = 2020 + delta // 12
    return str(year) + "-" + str(month)


def to_date_report(delta: int) -> str:
    """Return a date in the format of the date_report based on the number of months
     passes after 2020-01

        Preconditions:
          - delta > 0

        delta: The change in months after 2020-01
    """
    # Convert number of months into date, in date_report format
    month = delta % 12 + 1
    year = 2020 + delta // 12
    return "25-" + str(month) + "-" + str(year)
