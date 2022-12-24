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


import logging
import math
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Union

import colorcet
from pandas import to_datetime
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import functions as F


class Explorer(metaclass=ABCMeta):
    """Abstract base class used for EDA

    Methods that must be overridden are:
        make_hist
        make_box
        make_bar
        null_sort
        null_filter

        *see method docstrings for functionality and output signatures
    """

    @abstractmethod
    def make_bar(self, col: Union[List, str]):
        """Makes grouping-count for each column

        Args:
            col: list/name of columns to count for

        Returns:
            groupby pandas dataframe
        """
        pass

    @abstractmethod
    def make_box(self, cols: Union[List, str]) -> Dict:
        """Generates quartiles for box-whisker plotting


        Args:
            cols: list/string of column(s) to plot

        Output must be a dictionary indexed by:
        column
            | target_value
                | quantile

            *if target_values are categorical

        OR
        column
            | quantile

        Example output for binary target:
         {
            "num1": {
                "0": {
                    "q00": 803.0476168393415,
                    "q01": 803.0476168393415,
                    "q25": 907.2920695654886,
                    "q50": 959.0112145260407,
                    "q75": 1059.4072543931134,
                    "q99": 1197.3055442349162,
                    "q100": 1197.3055442349162
                },
                "1": {
                    "q00": 769.4188139185309,
                    "q01": 769.4188139185309,
                    "q25": 932.2965098911496,
                    "q50": 1003.9672686831474,
                    "q75": 1048.7928364881604,
                    "q99": 1308.6366589738643,
                    "q100": 1308.6366589738643
                }
            },
        }

        Quantiles for 0.00, 0.01, 0.25, 0.50, 0.75, 0.99 and 1.00 must be specified

        Returns:
            quantile_dict

        """
        pass

    @abstractmethod
    def make_hist(self, cols: Union[List, str], bins=None, n_bins=None) -> Dict:
        """Generates histogram for each column, split by target if type categorical/binary

        Args:
            cols: list/string of column(s) to generate histograms for
            bins: list of bin-edges
            n_bins: number of bins for histogram

        Raises:
            AttributeError if both bins and n_bins are defined

        Returns:
            dictionary of (bins, edges) indexed by column name
        """
        pass

    @abstractmethod
    def null_filter(self, data, _filter: str, p: float) -> pyspark.sql.DataFrame:
        """Filters columns with a percentage p missing or less"""
        pass

    @abstractmethod
    def null_sort(self, data, sort) -> pyspark.sql.DataFrame:
        """Sorts columns by percentage missing values"""
        pass

    @abstractmethod
    def plot_heatmap_missing(self, cols: list) -> None:
        """Plots a heatmap of correlation of missing values

        Args:
            cols: list of columns to plot missing correlation, if not specified, uses full dataset

        """
        pass

    def _validate_target_type(self, target):
        '''Helper method to ensure target is one of "binary", "categorical" or "continuous"'''
        if target not in ("binary", "categorical", "continuous"):
            raise ValueError(
                f'target must be one of "binary", "categorical" or "continuous", current target is {target}'
            )

    def plot_hist(self, cols: Union[List, str], bins=None, n_bins=None):
        """Plots histogram using output of make_hist function

        Args:
            cols: list/name of column to show histogram for
            bins: list of bin edges
            n_bins: number of bins

        Raises:
            AttributeError if both bins and n_bins are specified

        """

        if bins is not None and n_bins is not None:
            raise AttributeError("Specify either bins or n_bins, not both")

        if isinstance(cols, str):
            cols = [cols]

        hist_dict = self.make_hist(cols=cols, bins=bins, n_bins=n_bins)
        # single-column
        if len(hist_dict.keys()) == 1:
            col = list(hist_dict.keys())[0]
            fig = go.Figure()
            for idx, target_val in enumerate(self.target_vals):
                fig.add_trace(
                    go.Bar(
                        y=hist_dict[col][target_val]["bins"],
                        x=hist_dict[col][target_val]["edges"],
                        name=f"Target: {target_val}",
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
                            name=f"Target val: {target_val}",
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

    def plot_box(self, cols: Union[List, str] = None):
        """Plots quartiles using the make_box function

        Args:
            cols: list of columns to plot for, if not specified defaults to all

        """

        if not cols:
            cols = self.data.columns

        quantile_dict = self.make_box(cols)
        if self.target_type in ("binary", "categorical"):
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
                            name=f"Target: {target_val}",
                            marker=dict(color=colorcet.glasbey_dark[idx]),
                        )
                    )
                fig.update_layout(autosize=False, width=800, height=800, title=col)
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
                                name=f"Target: {target_val}",
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

        else:
            if len(quantile_dict.keys()) == 1:
                col = list(quantile_dict.keys())[0]
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        q1=[quantile_dict[col]["q25"]],
                        median=[quantile_dict[col]["q50"]],
                        q3=[quantile_dict[col]["q75"]],
                        x=[col],
                        upperfence=[quantile_dict[col]["q99"]],
                        lowerfence=[quantile_dict[col]["q01"]],
                        name=f"Column: {col}",
                        marker=dict(color=colorcet.glasbey_dark[0]),
                    )
                )
                fig.update_layout(autosize=False, width=800, height=800, title=col)
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
                    fig.add_trace(
                        go.Box(
                            q1=[quantile_dict[col]["q25"]],
                            median=[quantile_dict[col]["q50"]],
                            q3=[quantile_dict[col]["q75"]],
                            x=[col],
                            upperfence=[quantile_dict[col]["q99"]],
                            lowerfence=[quantile_dict[col]["q01"]],
                            name=f"Column: {col}",
                            marker=dict(color=colorcet.glasbey_dark[0]),
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
        """Scatter plot between two columns

        Args:
            col1: column on X-axis
            col2: column on Y-axis

        """
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

    def plot_bar(self, cols: Union[List, str]):
        grouped = self.make_bar(cols)

        if self.target_type in ("binary", "categorical"):
            self._plot_bar_categorical_target(cols, grouped)

        if self.target_type in ("continuous"):
            self._plot_bar_continuous_target(cols, grouped)

    def _plot_bar_categorical_target(self, cols, grouped):
        fig = go.Figure()
        for col in cols:
            for target_val in grouped[self.target].unique():
                fig.add_trace(
                    go.Bar(
                        x=grouped[col].unique(),
                        y=grouped.loc[grouped[self.target] == target_val, "count"],
                        name=f"Target: {target_val}",
                    )
                )
        fig.show()

    def _plot_bar_continuous_target(self, cols, grouped):
        fig = go.Figure()
        for col in cols:
            fig.add_trace(
                go.Bar(
                    x=grouped[col].unique(),
                    y=grouped["count"],
                )
            )

        fig.show()

    def plot_corr_map(self, cols: list = None):

        if not cols:
            cols = self.data.columns
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
        """Spark-specific implementation of Explorer


        Args:
            data: pyspark dataframe
            target: name of target column
            target_type: one of "continuous", "categorical" or "binary"
            date_cols: either a list of ['year', 'month'] OR a single column
            date_format: one of "american", "julian" or "iso" with the dash/dot variations

        Raises:
            ValueError: if target-type is not valid

        """

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
        return list(data.select(target).distinct().toPandas()[target].values)

    def make_bar(self, cols):
        """Makes grouping-count for each column

        Args:
            col: list/name of columns to count for

        Returns:
            groupby pandas dataframe
        """
        if self.target_type in ("binary", "categorical"):
            return self.data.groupBy(*[self.target, *cols]).count().toPandas()
        else:
            return self.data.groupby(*cols).count().toPandas()

    def make_box(self, cols):
        """Generates quartiles for box-whisker plotting


        Args:
            cols: list/string of column(s) to plot

        Output must be a dictionary indexed by:
        column
            | target_value
                | quantile

            *if target_values are categorical

        OR
        column
            | quantile

        Example output for binary target:
         {
            "num1": {
                "0": {
                    "q00": 803.0476168393415,
                    "q01": 803.0476168393415,
                    "q25": 907.2920695654886,
                    "q50": 959.0112145260407,
                    "q75": 1059.4072543931134,
                    "q99": 1197.3055442349162,
                    "q100": 1197.3055442349162
                },
                "1": {
                    "q00": 769.4188139185309,
                    "q01": 769.4188139185309,
                    "q25": 932.2965098911496,
                    "q50": 1003.9672686831474,
                    "q75": 1048.7928364881604,
                    "q99": 1308.6366589738643,
                    "q100": 1308.6366589738643
                }
            },
        }

        Quantiles for 0.00, 0.01, 0.25, 0.50, 0.75, 0.99 and 1.00 must be specified

        Returns:
            quantile_dict

        """
        if self.target_type in ("binary", "categorical"):
            return self._make_box_categorical_target(cols)
        else:
            return self._make_box_continuous_target(cols)

    def _make_box_categorical_target(self, cols):
        quantile_dict = {}
        for col in cols:
            # q00, q01, q25, q50, q75, q99, q100 =
            target_q = (
                self.data.groupBy(self.target)
                .agg(
                    F.percentile_approx(
                        col, [0.00, 0.01, 0.25, 0.50, 0.75, 0.99, 1.00], 1000
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

        return quantile_dict

    def _make_box_continuous_target(self, cols):
        for col in cols:
            q00, q01, q25, q50, q75, q99, q100 = (
                self.data.select(
                    F.percentile_approx(
                        "num1", [0.00, 0.01, 0.25, 0.50, 0.75, 0.99, 1.00]
                    ).alias("percentiles")
                )
                .collect()[0]
                .percentiles
            )

            quantile_dict[col] = {
                "q00": q00,
                "q01": q01,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q99": q99,
                "q100": q100,
            }

        return quantile_dict

    def make_hist(self, cols, n_bins=None, bins=None):
        """Generates histogram for each column, split by target if type categorical/binary

        Args:
            cols: list/string of column(s) to generate histograms for
            bins: list of bin-edges
            n_bins: number of bins for histogram

        Raises:
            AttributeError if both bins and n_bins are defined

        Returns:
            dictionary of (bins, edges) indexed by column name
        """
        if n_bins and bins:
            raise ValueError("Specify one of n_bins or bins")

        if not isinstance(bins, list):
            raise TypeError("bins must be a list of integers or long")

        col_dict = {}
        for col in cols:
            target_dict = {}
            for target_val in self.target_vals:
                target_dict[target_val] = {"bins": None, "edges": None}
                target_dict[target_val]["edges"], target_dict[target_val]["bins"] = (
                    dd.filter(F.col(self.target) == int(target_val))
                    .select(col)
                    .rdd.flatMap(lambda x: x)
                    .histogram(n_bins if n_bins else bins)
                )

                target_dict[target_val]["edges"] = target_dict[target_val]["edges"][
                    :-1
                ]  # off-by-one

            col_dict[col] = target_dict

        return col_dict

    @staticmethod
    def null_sort(data, sort=None):
        """Sorts columns by percentage missing values"""
        if sort is None:
            return data

        if sort not in ["ascending", "descending"]:
            raise ValueError('sort must be on of "ascending" or "descending"')

        counts = data.select(
            *[F.count(col).alias(col) for col in data.columns]
        ).toPandas()

        counts.index = ["cnt"]
        cols = counts.columns
        vals = counts.values[0]

        series = pd.Series(data=vals, index=cols)
        new_col_order = []
        if sort == "ascending":
            for col_id in np.argsort(series).values:
                new_col_order.append(data.columns[col_id])

            return data.select(*new_col_order)

        elif sort == "descending":
            for col_id in np.flipud(np.argsort(series.values)):
                new_col_order.append(data.columns[col_id])

            return data.select(*new_col_order)

    @staticmethod
    def null_filter(data, _filter=None, p=0, n=0):
        new_col_order = data.columns
        if _filter == "top":
            new_col_order = []
            len_df = data.count()
            if p:
                for idx, col in enumerate(data.columns):
                    if count_rows_values(data)[idx] / len_df >= p:
                        new_col_order.append(col)

            if n:
                raise NotImplementedError

        elif _filter == "bottom":
            new_col_order = []
            len_df = data.count()
            if p:
                for idx, col in enumerate(data.columns):
                    if count_rows_values(data)[idx] / len_df <= p:
                        new_col_order.append(col)

            if n:
                raise NotImplementedError

        return data.select(*new_col_order)

    @staticmethod
    def count_rows_values(data: pyspark.sql.DataFrame):
        """Does pandas equivalent of .count(axis='rows') but for pyspark"""
        counts = (
            data.select(*[F.count(col).alias(col) for col in data.columns])
            .toPandas()
            .values[0]
        )

        return pd.Series(index=data.columns, data=counts)

    def plot_heatmap_missing(self, cols: list = None):
        """Plots a heatmap of correlation of missing values

        Args:
            cols: list of columns to plot missing correlation, if not specified, uses full dataset

        """
        if not cols:
            cols = self.data.columns

        data = self.data.select(*cols)
        data = self.null_filter(data)
        data = self.null_sort(data, sort=None)

        col_counts = self.count_rows_values(data)
        if isinstance(data, pyspark.sql.DataFrame):
            data = data.toPandas()

        data_small = data[col_counts.index]
        corr_mat = data_small.isnull().corr()
        if corr_mat.sum().sum() == 0:
            logging.info("No missing values or empty dataframe")
            return
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
                width=800,
                height=800,
                title="Missing Value Correlations",
            ),
        )

        fig.show()


if __name__ == "__main__":
    from pyspark.sql import DataFrame, SparkSession

    spark = SparkSession.builder.appName("eda").getOrCreate()
    N = 10000
    df = pd.DataFrame(
        {
            "id": ["A"] * N,
            "year": [2023] * N,
            "month": np.random.randint(1, 12, size=N),  # not real obviously
            "num1": np.random.normal(100, N, size=N),
            "num2": np.random.normal(10, N, size=N),
            "cat1": np.random.choice(["a", "b", "c", None], size=N),
            "cat2": np.random.choice(["A", "B", "C", None], size=N),
            "bin_target": np.random.choice([0, 1], size=N, p=[0.2, 0.8]),
            "multi_target": np.random.choice(
                [-1, 0, 1], size=N, replace=True, p=[0.2, 0.5, 0.3]
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

    s.plot_hist(["num1", "num2"], bins=[int(i) for i in range(0, 1100, 100)])

    s.plot_box(["num1", "num2"])

    s.plot_corr_map(cols=["num1", "num2"])

    s.plot_bar(["cat1"])

    s.plot_heatmap_missing()

    s.plot_corr_scatter("num1", "num2")
