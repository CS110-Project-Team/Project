"""CSC110: Final Project

Description
===============================

This file uses various data sets scavenged from the internet and outputs four choropleth maps
that graph the number of cases, grocery items, receipts from full-service restaurants, and
consumer price index of utensils. Each choropleth map animates each monthly data row
starting from January 2020 to September 2021 and illustrates whether if there was a
change with our complements and substitutes during the peak COVID-19 cases spike. We
also used a linear regression model on our data sets to predict our various data sets
until January 2021. Hence our four choropleth graphs map data for 24 months.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of TAs and instructors
of CSC110 at the University of Toronto St. George campus. All forms of distribution of
this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2021 Herman Vuong, and Sophie Xu.
"""
import json
from urllib.request import urlopen
import plotly.express as plotly
import pandas
import python_ta

from data_wrangling import clean_food_price_data, clean_case_data, add_id_column
from linear_regression import correlation, correlation2, predict_four_month


def plot_choropleth(dataset: pandas.DataFrame, colour: str, scale: list,
                    animation: str, title: str) -> None:
    """Display the choropleth plot of the given dataset.

        Preconditions:
          - len(dataset) != 0

        colour: name of predefined colour scale for the map
        scale: list containing the range of values for the scale
        animation: variable in dataframe to create a slider for
        title: title of the graph
    """
    figure = plotly.choropleth(dataset, locations='id', geojson=provinces, color='VALUE',
                               scope='north america', color_continuous_scale=colour,
                               animation_frame=animation, hover_name='GEO', range_color=scale,
                               title=title)
    figure.update_geos(fitbounds='locations', visible=False)
    figure.show()


if __name__ == '__main__':
    python_ta.check_all(config={
        'extra-imports': [True],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    # Load Data
    date = pandas.read_csv("date.csv")
    receipt = pandas.read_csv("receipt.csv")
    cpi_utensil = pandas.read_csv("utensil.csv")
    cpi_food = pandas.read_csv("food.csv")
    cases = pandas.read_csv('https://raw.githubusercontent.com/'
                            'ccodwg/Covid19Canada/master/timeseries_prov'
                            '/cases_timeseries_prov.csv')
    # Loading province geojson data
    with urlopen(
            'https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/'
            'public/data/canada.geojson') \
            as response:
        provinces = json.load(response)

    # Creating ids for provinces
    province_ids = {}
    # Mutate geojson data
    for i in range(len(provinces['features'])):
        province_ids[provinces['features'][i]['properties']['name']] \
            = provinces['features'][i]['properties']['cartodb_id']
        provinces['features'][i]['id'] = provinces['features'][i]['properties']['cartodb_id']

    # Cleaning data
    receipt = clean_food_price_data(name="receipt_data", dataset=receipt,
                                    filters="Full-service restaurants [722511]", scale=1)
    food = clean_food_price_data(name="food_data", dataset=cpi_food, filters="Food", scale=1)
    utensil = clean_food_price_data(name="utensil_data", dataset=cpi_utensil,
                                    filters="Non-electric kitchen utensils, tableware and cookware",
                                    scale=1)
    cases = clean_case_data(name="cases_terr_data", dataset=cases)

    # Predicting data for all data sets for the next four months
    receipt_predict = predict_four_month(receipt, date, case=False)
    food_predict = predict_four_month(food, date, case=False)
    utensil_predict = predict_four_month(utensil, date, case=False)
    cases_predict = predict_four_month(cases, date, case=True)

    # Combine the original data set with the predicted data set
    receipt_full = pandas.concat([receipt, receipt_predict])
    food_full = pandas.concat([food, food_predict])
    utensil_full = pandas.concat([utensil, utensil_predict])
    cases_full = pandas.concat([cases, cases_predict])

    # Adding id column to all data sets
    receipt_final = add_id_column(dataset=receipt_full, province_to_id=province_ids)
    food_final = add_id_column(dataset=food_full, province_to_id=province_ids)
    utensil_final = add_id_column(dataset=utensil_full, province_to_id=province_ids)
    cases_final = add_id_column(dataset=cases_full, province_to_id=province_ids)

    # Plotting data onto choropleth maps
    # plot_choropleth(dataset=utensil_final, colour='pinkyl', scale=[0, 110],
    #                 animation='REF_DATE', title='Consumer Price Index of Utensils')
    # plot_choropleth(dataset=food_final, colour='greens', scale=[0, 160],
    #                 animation='REF_DATE', title='Consumer Price Index of Food')
    # plot_choropleth(dataset=receipt_final, colour='blues', scale=[0, 1_500_000],
    #                 animation='REF_DATE', title='Number of Receipts for Full-Service Restaurants')
    # plot_choropleth(dataset=cases_final, colour='reds', scale=[0, 3500],
    #                 animation='date_report', title='Number of Active Cases')

    # Display Scatter Plots
    # correlation2(receipt, utensil, cases, food)

    # See Individual Plots
    # correlation(receipt, food, "receipt")
    # correlation(utensil, food, "utensil")
    correlation(cases, food, "cases")
