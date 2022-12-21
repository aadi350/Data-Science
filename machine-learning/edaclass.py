#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import colorcet
import numpy as np
import pandas as pd
import pyspark
from pandas import to_datetime
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import functions as F


class DateFormatError(Exception):
    """Custom Exception thrown for date types not supported"""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(self.message)


class SparkDateUtil(object):
    """Used to create a consistent year-month column in pyspark dataframes

    Supported arguments to create_yearmonth are:
        df: a pyspark dataframe
        date_cols: either a list ('year', 'month')
        format: a format specified in _DATE_FORMAT_MAP OR 'list' if date_cols is a list

    Example usage:
        df = # pyspark dataframe
        df = SparkDateUtil.create_yearmonth(df, date_cols=['year', 'month'], format='list')

    Raises:
        DateFormatError: if the number of date-columns is more than 3
        ValueError: thrown from the spark to_date method

    """

    _DATE_FORMAT_MAP = {
        "american": "%m%d%y",
        "julian": "%Y%j",
        "iso": "%y%m%d",
        "american_dash": "%m-%d-%y",
        "american_fwslash": "%m/%d/%y",
        "julian_dash": "%Y-%j",
        "julian_fwslash": "%Y/%j",
        "iso_dash": "%y-%m-%d",
        "iso_fwslash": "%y/%m/%d",
    }

    @staticmethod
    def create_yearmonth(df, date_cols: Union[List, str], format: str):
        if format == "list":
            if len(date_cols) > 3:
                raise DateFormatError("Year, Month and Day alone supported")
            if len(date_cols) == 2:
                # year and month OR Year and Month
                if "year" in date_cols and "month" in date_cols:
                    df = SparkDateUtil._create_spark_yearmonth_from_list(df, date_cols)
                if "Year" in date_cols and "Month" in date_cols:
                    df = SparkDateUtil._create_spark_yearmonth_from_list(df, date_cols)

        else:
            df = df.withColumn(
                "year_month",
                F.to_date(
                    F.col(date_cols), format=SparkDateUtil._DATE_FORMAT_MAP[format]
                ),
            )

        return df

    @staticmethod
    def _create_spark_yearmonth_from_list(df, date_cols):
        if len(date_cols) == 2:
            return df.withColumn(
                "year_month", F.expr(f"make_date({date_cols[0]}, {date_cols[1]}, 1)")
            )
        elif len(date_cols) == 3:
            return df.withColumn(
                "year_month",
                F.expr(f"make_date({date_cols[0]}, {date_cols[1]}, {date_cols[2]})"),
            )


class PandasDateUtil(object):
    """Used to create a consistent year-month column in pandas dataframes

    Supported arguments to create_yearmonth are:
        df: a Pandas dataframe
        date_cols: either a list ('year', 'month')
        format: a format specified in _DATE_FORMAT_MAP OR 'list' if date_cols is a list

    Example usage:
        df = # pandas dataframe
        df = PandasDateUtil.create_yearmonth(df, date_cols=['year', 'month'], format='list')

    Raises:
        DateFormatError: if the number of date-columns is more than 3
        ValueError: thrown from pandas in the to_datetime method

    """

    _DATE_FORMAT_MAP = {
        "american": "%m%d%Y",
        "julian": "%Y%j",
        "iso": "%Y%m%d",
        "american_dash": "%m-%d-%Y",
        "american_fwslash": "%m/%d/%Y",
        "julian_dash": "%Y-%j",
        "julian_fwslash": "%Y/%j",
        "iso_dash": "%Y-%m-%d",
        "iso_fwslash": "%Y/%m/%d",
    }

    @staticmethod
    def create_yearmonth(df, date_cols: Union[List, str], format: str):
        if format == "list":
            if len(date_cols) > 3:
                raise DateFormatError("Year, Month and Day alone supported")
            if len(date_cols) == 2:
                # year and month OR Year and Month
                if "year" in date_cols and "month" in date_cols:
                    return df.assign(
                        year_month=to_datetime(
                            dict(year=df.year, month=df.month, day=1)
                        )
                    )
                elif "Year" in date_cols and "Month" in date_cols:
                    return df.assign(
                        year_month=to_datetime(
                            dict(year=df.Year, month=df.month, day=1)
                        )
                    )
        else:
            if PandasDateUtil._DATE_FORMAT_MAP.get(format, None) is not None:

                logging.debug(PandasDateUtil._DATE_FORMAT_MAP[format])
                return df.assign(
                    year_month=to_datetime(
                        df[date_cols],
                        format=PandasDateUtil._DATE_FORMAT_MAP.get(format, None),
                    )
                )
            else:
                return df.assign(year_month=to_datetime(df[date_cols]))


class Explorer(ABC):
    @abstractmethod
    def make_box(self, cols: Union[List, str]) -> Dict:
        pass

    @abstractmethod
    def make_hist(self, cols: Union[List, str], bins=None) -> Dict:
        pass

    def _validate_target_type(self, target_type):
        if target_type not in ("binary", "continuous", "categorical"):
            raise ValueError(f"target type not supported: {target_type}")

    def plot_hist(self, cols: Union[List, str], bins=None):
        if isinstance(cols, str):
            cols = [cols]

        hist_dict = self.make_hist(cols=cols)
        # single-column
        if len(hist_dict.keys()) == 1:
            col = list(hist_dict.keys())[0]
            fig = go.Figure()
            for idx, target_val in enumerate(self.target_vals):
                fig.add_trace(
                    go.Bar(
                        y=hist_dict[col][target_val]["bins"],
                        x=hist_dict[col][target_val]["edges"],
                        name=target_val,
                        marker=dict(color=colorcet.glasbey_dark[idx]),
                    )
                )
                fig.update_layout(autosize=False, width=800, height=800, title=col)
        else:
            num_cols = len(list(hist_dict.keys()))
            fig = make_subplots(
                rows=math.ceil(num_cols / 3),
                cols=3,
                subplot_titles=list(hist_dict.keys()),
            )
            row_num = 1
            col_num = 1
            for col in hist_dict.keys():
                for idx, target_val in enumerate(self.target_vals):
                    fig.add_trace(
                        go.Bar(
                            y=hist_dict[col][target_val]["bins"],
                            x=hist_dict[col][target_val]["edges"],
                            name=target_val,
                            marker=dict(color=colorcet.glasbey_dark[idx]),
                        ),
                        row=row_num,
                        col=col_num,
                    )

                col_num += 1
                if col_num > 3:
                    row_num += 1
                    col_num = 1

            fig.update_layout(
                autosize=False, width=1200, height=math.ceil(num_cols / 3) * 400
            )

        fig.show()

    def plot_box(self, cols: Union[List, str]):
        quantile_dict = self.make_box(cols)
        if len(quantile_dict.keys()) == 1:
            col = list(quantile_dict.keys())[0]
            fig = go.Figure()
            for idx, target_val in enumerate(self.target_vals):
                fig.add_trace(
                    go.Box(
                        q1=[quantile_dict[col][target_val]["q25"]],
                        median=[quantile_dict[col][target_val]["q50"]],
                        q3=[quantile_dict[col][target_val]["q75"]],
                        x=[target_val],
                        upperfence=[quantile_dict[col][target_val]["q99"]],
                        lowerfence=[quantile_dict[col][target_val]["q01"]],
                        name=target_val,
                        marker=dict(color=colorcet.glasbey_dark[idx]),
                    )
                )

            fig.show()

        else:
            num_cols = len(list(quantile_dict.keys()))
            fig = make_subplots(
                rows=math.ceil(num_cols / 3),
                cols=3,
                subplot_titles=list(quantile_dict.keys()),
            )
            row_num = 1
            col_num = 1
            for col in quantile_dict.keys():
                for idx, target_val in enumerate(self.target_vals):
                    fig.add_trace(
                        go.Box(
                            q1=[quantile_dict[col][target_val]["q25"]],
                            median=[quantile_dict[col][target_val]["q50"]],
                            q3=[quantile_dict[col][target_val]["q75"]],
                            x=[target_val],
                            upperfence=[quantile_dict[col][target_val]["q99"]],
                            lowerfence=[quantile_dict[col][target_val]["q01"]],
                            name=target_val,
                            marker=dict(color=colorcet.glasbey_dark[idx]),
                        ),
                        row=row_num,
                        col=col_num,
                    )

                col_num += 1
                if col_num > 3:
                    row_num += 1
                    col_num = 1

            fig.update_layout(
                autosize=False, width=1200, height=math.ceil(num_cols / 3) * 400
            )
            fig.show()

    def plot_corr_scatter(self, col1, col2):
        corr = self.data.select(*[col1, col2]).stat.corr("num1", "num2")
        data = self.data.select(*[col1, col2, self.target]).toPandas()
        self._plot_corr_scatter(col1, col2, data)

    def _plot_corr_scatter(self, col1, col2, data):
        if len(data) > 10000:
            logging.warning("Sampling done, data too large")
            data = data.sample(10000)

        fig = px.scatter(
            data,
            col1,
            col2,
            marginal_x="histogram",
            marginal_y="histogram",
            color=data[self.target],  # HACK need to set target to astype(category)
        )

        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(title=col1),
            yaxis=dict(title=col2),
            title=f"{col2} against {col1}",
        )
        fig.show()

    @abstractmethod
    def make_bar(self, col: Union[List, str]):
        pass

    def plot_bar(self, col: Union[List, str]):
        grouped = self.make_bar(col)
        fig = go.Figure()
        for target_val in grouped[self.target].unique():
            fig.add_trace(
                go.Bar(
                    x=grouped[col].unique(),
                    y=grouped.loc[grouped[self.target] == target_val, "count"],
                    name=f"Target: {target_val}",
                )
            )
        fig.show()

    def plot_corr_map(self, cols: list):
        import pandas
        import pyspark

        pio.templates.default = "plotly_white"
        if isinstance(self.data, pyspark.sql.DataFrame):
            data = self.data[cols].toPandas()
        else:
            data = self.data[cols]

        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        rLY = corr.mask(mask)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.mask(mask),
                x=corr.columns,
                y=corr.columns,
                colorscale=colorcet.coolwarm,
                zmin=-1,
                zmax=1,
            ),
            layout=go.Layout(
                autosize=False,
                height=800 if len(corr.columns) < 4 else len(corr.columns) * 100,
                width=800 if len(corr.columns) < 4 else len(corr.columns) * 100,
            ),
        )
        fig.show()

    @abstractmethod
    def null_filter(self, data, _filter: str, p: float):
        pass

    @abstractmethod
    def null_sort(self, data, sort):
        pass

    @abstractmethod
    def plot_heatmap_missing(self, data):
        """These are too different to abstract away with common plotting functions"""
        pass


class SparkExplorer(
    Explorer,
):
    def __init__(
        self,
        data: pyspark.sql.DataFrame,
        target: str,
        target_type: str,
        date_cols: Union[List, str] = None,
        date_format: str = None,
    ):

        self._validate_target_type(target_type)
        self.target_vals = None

        if date_cols and date_format:
            data = SparkDateUtil.create_yearmonth(
                data, date_cols=date_cols, format=date_format
            )

        if target_type in ("binary", "categorical"):
            self.target_vals = self._get_target_vals(data, target)

        self.data = data
        self.target = target
        self.target_type = target_type

    def _get_target_vals(self, data, target):
        return data.select(target).distinct().toPandas()[target].values

    def make_bar(self, cols):
        return self.data.groupBy([self.target, cols]).count().toPandas()

    def make_box(self, cols):
        quantile_dict = {}
        for col in cols:
            # q00, q01, q25, q50, q75, q99, q100 =
            target_q = (
                self.data.groupBy(self.target)
                .agg(
                    F.percentile_approx(
                        col, [0.00, 0.01, 0.25, 0.50, 0.75, 0.99, 1.00], 1
                    ).alias("percentiles")
                )
                .collect()
            )

            target_dict = {}
            for target in target_q:
                target_val = target[self.target]
                q00, q01, q25, q50, q75, q99, q100 = target.percentiles
                target_dict[target_val] = {
                    "q00": q00,
                    "q01": q01,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                    "q99": q99,
                    "q100": q100,
                }
            quantile_dict[col] = target_dict

        # returns dictionary indexed by column -> target_value -> quantile

        return quantile_dict

    def make_hist(self, cols, n_bins=None, bins=None):
        print(f"bins: {bins}, n_bins: {n_bins}")
        col_dict = {}
        if self.target_type in ("binary", "categorical"):
            for col in cols:
                print(col)
                target_dict = {}
                for target_val in self.target_vals:
                    target_dict[target_val] = self.data.filter(
                        F.col(self.target) == target_val
                    ).select(
                        col
                    )  # .rdd.flatMap(lambda x: x).histogram(bins if bins else n_bins))

                col_dict[col] = target_dict

            return col_dict

        elif self.target_type in ("continuous"):
            for col in cols:
                col_dict[col] = (
                    self.data.select(col)
                    .rdd.flatMap(lambda x: x)
                    .histogram(n_bins if n_bins else bins)
                )

        return col_dict

    def null_filter(self, data):
        return data

    def null_sort(self, data):
        return data

    def count_rows_values(self, data: pyspark.sql.DataFrame):
        counts = (
            data.select(*[F.count(col).alias(col) for col in data.columns])
            .toPandas()
            .values[0]
        )

        return pd.Series(index=data.columns, data=counts)

    def plot_heatmap_missing(self, cols: list):
        data = self.data.select(*cols)
        data = self.null_filter(data)
        data = self.null_sort(data)

        col_counts = self.count_rows_values(data)

        if isinstance(data, pyspark.sql.DataFrame):
            data = data.toPandas()

        data_small = data[col_counts.index]
        corr_mat = data_small.isnull().corr()
        mask = np.zeros_like(corr_mat)
        mask[np.triu_indices_from(mask)] = True

        fig = go.Figure(
            go.Heatmap(
                z=corr_mat.values,
                x=corr_mat.columns,
                y=corr_mat.columns,
                zmin=-1,
                zmax=1,
                colorscale=colorcet.coolwarm,
            ),
            layout=go.Layout(
                autosize=False,
                width=1200,
                height=1200,
                title="Missing Value Correlations",
            ),
        )

        fig.show()


if __name__ == "__main__":
    from pyspark.sql import DataFrame, SparkSession

    spark = SparkSession.builder.appName("eda").getOrCreate()
    df = pd.DataFrame(
        {
            "id": ["A"] * 100,
            "year": [2023] * 100,
            "month": np.linspace(1, 100, 100),  # not real obviously
            "num1": np.random.normal(1000, 100, size=100),
            "num2": np.random.normal(10, 100, size=100),
            "cat1": np.random.choice(["a", "b", "c"], size=100),
            "cat2": np.random.choice(["A", "B", "C"], size=100),
            "bin_target": np.random.choice([0, 1], size=100, p=[0.2, 0.8]),
            "multi_target": np.random.choice(
                [-1, 0, 1], size=100, replace=True, p=[0.2, 0.5, 0.3]
            ),
        }
    )

    dd = spark.createDataFrame(df)

    # testing spark explorer
    s = SparkExplorer(
        data=dd,
        target="bin_target",
        target_type="binary",
    )
