"""
This file contains the functions for constructing and using linear regression models
"""
import pandas
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot
from data_wrangling import to_date_report, to_ref_date


def scatter_plot(x_dataset: pandas.Series, y_dataset: pandas.Series, title: str, x_label: str,
                 y_label: str) -> None:
    """Display the scatter plot of the given dataset with the line of best fit.
    """
    predictor = x_dataset["VALUE"].values.reshape(-1, 1)
    response = y_dataset["VALUE"].values.reshape(-1, 1)

    # Show Scatter Plot
    plot.scatter(predictor, response, color="blue")
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.show()

    # Show Line of Best Fit
    line = model(x_dataset, y_dataset)
    slope, intercept = line[1][0], line[2][0]
    maximum = max(x_dataset["VALUE"].tolist())
    minimum = min(x_dataset["VALUE"].tolist())
    x, y = [minimum, maximum], [intercept + (slope * minimum), intercept + (slope * maximum)]
    plot.plot(x, y, color="pink", linewidth=3)


def scatter_plot2(x_dataset: pandas.Series, y_dataset: pandas.Series, title: str, legend: str, colour: str) -> None:
    """Display and accumulate the scatter plot of the given datasets with the lines of best fit.
    """
    predictor = x_dataset["VALUE"].values.reshape(-1, 1)
    response = y_dataset["VALUE"].values.reshape(-1, 1)

    # Show Scatter Plot
    plot.scatter(predictor, response, color=colour, label=legend)
    plot.legend(loc="upper left")
    plot.title(title)
    plot.xlabel("Count")
    plot.ylabel("Food Price")
    plot.show()

    # Show Line of Best Fit
    line = model(x_dataset, y_dataset)
    slope, intercept = line[1][0], line[2][0]
    maximum = max(x_dataset["VALUE"].tolist())
    minimum = min(x_dataset["VALUE"].tolist())
    x, y = [minimum, maximum], [intercept + (slope * minimum), intercept + (slope * maximum)]
    plot.plot(x, y, color=colour, linewidth=3)


def correlation(dataset: pandas.Series, food: pandas.Series, predictor: str) -> None:
    """Display the scatter plot to show correlation between the given predictor and food price.

    Precondition:
    - predictor in {"receipt", "utensil", "cases"}
    """
    # Call scatter plot for Receipt Dataset
    if predictor == "receipt":
        scatter_plot(x_dataset=dataset, y_dataset=food,
                     title="Impact of Number of Receipts for Full Service Restaurants on Food Price",
                     x_label="Number of Receipts", y_label="Food Price")
    # Call scatter plot for Utensil Dataset
    elif predictor == "utensil":
        scatter_plot(x_dataset=dataset, y_dataset=food,
                     title="Impact of Price of Kitchen Utensil on Food Price",
                     x_label="Kitchen Utensil Price", y_label="Food Price")
    # Call scatter plot for Cases Dataset
    elif predictor == "cases":
        scatter_plot(x_dataset=dataset, y_dataset=food,
                     title="Impact of COVID-19 Cases on Food Price",
                     x_label="Number of COVID-19 Cases", y_label="Food Price")


def correlation2(receipt: pandas.Series, utensil: pandas.Series, cases: pandas.Series, food: pandas.Series) -> None:
    """Display the scatter plot to show correlation between the datasets and food price.
    """
    # Call and combine all three scatter plots
    scatter_plot2(x_dataset=receipt, y_dataset=food,
                  title="Impact of Number of Receipts for Full Service Restaurants on Food Price",
                  colour="pink", legend="Receipt")
    scatter_plot2(x_dataset=utensil, y_dataset=food,
                  title="Impact of Price of Kitchen Utensil on Food Price",
                  colour="orange", legend="Utensil Index")
    scatter_plot2(x_dataset=cases, y_dataset=food,
                  title="Impact of COVID-19 Cases on Food Price",
                  colour="red", legend="Cases")


def model(x_dataset: pandas.Series, y_dataset: pandas.Series) -> [float, float, float]:
    """Return the coefficient of determination, slope and intercept of the linear regression model built using
    the date as predictor and the given dataset as response variable.
    """
    predictor = x_dataset["VALUE"].values.reshape(-1, 1)
    response = y_dataset["VALUE"].values.reshape(-1, 1)
    model1 = LinearRegression()
    model1.fit(predictor, response)
    return [model1.score(predictor, response), model1.coef_, model1.intercept_]


def predict(slope: float, intercept: float, x_dataset: pandas.DataFrame) -> list:
    """Return the predicted value based on the line of best fit.
    """
    # Extrapolate values using the slope and intercept of the line of best fit
    predictions_so_far = []
    for i, row in x_dataset.iterrows():
        prediction = intercept + (x_dataset.at[i, "VALUE"]) * slope
        predictions_so_far.append(prediction)
    return predictions_so_far


def predict_four_month(response: pandas.Series, date: pandas.Series, case: bool) -> pandas.DataFrame:
    """Return the predicted values of a given dataset using linear regression, for all province and territories for
        October, November, December in 2021 and January in 2022

    case: indicates the prediction is for cases
    """
    # Helper Variables for Prediction
    future_date = pandas.DataFrame(data=[["2021-10", 21], ["2021-11", 22], ["2021-12", 23], ["2022-01", 24]],
                                   columns=["REF_DATE", "VALUE"])
    province_name = {"Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador",
                     "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"}

    # Predict values for four Months
    predictions_so_far = []
    for province in province_name:
        y_dataset = response[response["GEO"] == province]
        line = model(date, y_dataset)
        slope, intercept = line[1][0], line[2][0]
        predictions = predict(slope, intercept, future_date)

        # Collecting predicted values into DataFrame
        index = 0
        for i, row in future_date.iterrows():
            if case:
                new_data = [to_date_report(future_date.at[i, "VALUE"]), province, float(predictions[index])]
            else:
                new_data = [to_ref_date(future_date.at[i, "VALUE"]), province, float(predictions[index])]
            predictions_so_far.append(new_data)
            index = index + 1
    if case:
        all_data = pandas.DataFrame(data=predictions_so_far, columns=["date_report", "GEO", "VALUE"])
    else:
        all_data = pandas.DataFrame(data=predictions_so_far, columns=["REF_DATE", "GEO", "VALUE"])
    return all_data
