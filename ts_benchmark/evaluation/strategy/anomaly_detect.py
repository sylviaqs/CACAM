# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """
    REQUIRED_CONFIGS = ["seed"]

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None


    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed(self.strategy_config.get("seed", 2021))

        model = model_factory()
        try:
            self.model = model
            if hasattr(model, "set_experiment_context"):
                model.set_experiment_context(
                    series_name=series_name,
                    model_name=model_factory.model_name,
                    strategy_name=self.strategy_config.get("strategy_name"),
                    metrics=self.evaluator.metric_names,
                    seed=self.strategy_config.get("seed"),
                )
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
            else:
                self.model.fit(train_data, train_label)  # 在训练数据上拟合模型

            end_fit_time = time.time()
            predict_labels, another = self.detect(test_data)
            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                remaining_length = len(actual_label) - len(predict_label)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )
                    another = np.pad(
                        another,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float)
                )
                if hasattr(model, "record_evaluation_result"):
                    model.record_evaluation_result(
                        ratio=ratio,
                        metrics=dict(zip(self.evaluator.metric_names, single_series_results)),
                        fit_time_sec=end_fit_time - start_fit_time,
                        inference_time_sec=end_inference_time - end_fit_time,
                    )

                inference_data = [predict_label, another]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )
                single_series_results += [
                    series_name,
                    end_fit_time - start_fit_time,
                    end_inference_time - end_fit_time,
                    ratio,
                    '',
                    '',
                    log_info,
                ]
                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list

    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]


class FixedDetectScore(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )

        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)


    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AnomalyDetectBoth(AnomalyDetect):
    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        fix_random_seed(self.strategy_config.get("seed", 2021))

        model = model_factory()
        try:
            self.model = model
            if hasattr(model, "set_experiment_context"):
                model.set_experiment_context(
                    series_name=series_name,
                    model_name=model_factory.model_name,
                    strategy_name=self.strategy_config.get("strategy_name"),
                    metrics=self.evaluator.metric_names,
                    seed=self.strategy_config.get("seed"),
                )
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)
            else:
                self.model.fit(train_data, train_label)

            end_fit_time = time.time()
            predict_labels, _ = self.model.detect_label(test_data)
            predict_scores, another = self.model.detect_score(test_data)

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}
            if not isinstance(predict_scores, dict):
                predict_scores = {"None": predict_scores}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_score = self._get_ratio_output(predict_scores, ratio)
                predict_label = self._pad_prediction(predict_label, len(actual_label), 0)
                predict_score = self._pad_prediction(predict_score, len(actual_label), 0.0)
                another_ratio = self._get_ratio_output(another, ratio)
                another_ratio = self._pad_prediction(another_ratio, len(actual_label), 0.0)

                single_series_results, log_info = self._evaluate_both(
                    actual_label.astype(float),
                    predict_label.astype(float),
                    predict_score.astype(float),
                )
                if hasattr(model, "record_evaluation_result"):
                    model.record_evaluation_result(
                        ratio=ratio,
                        metrics=dict(zip(self.evaluator.metric_names, single_series_results)),
                        fit_time_sec=end_fit_time - start_fit_time,
                        inference_time_sec=end_inference_time - end_fit_time,
                    )

                inference_data = [predict_label, predict_score, another_ratio]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )
                single_series_results += [
                    series_name,
                    end_fit_time - start_fit_time,
                    end_inference_time - end_fit_time,
                    ratio,
                    '',
                    '',
                    log_info,
                ]
                single_series_results_list.append(single_series_results)
        except Exception as e:
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list

    def _evaluate_both(
        self,
        actual: np.ndarray,
        predicted_label: np.ndarray,
        predicted_score: np.ndarray,
    ):
        evaluate_result = []
        log_info = ""
        for metric_info, metric_name, metric_func in zip(
            self.evaluator.metric,
            self.evaluator.metric_names,
            self.evaluator.metric_funcs,
        ):
            predicted = (
                predicted_score
                if metric_info["name"] in classification_metrics_score.__all__
                else predicted_label
            )
            try:
                evaluate_result.append(metric_func(actual, predicted))
            except Exception as e:
                evaluate_result.append(np.nan)
                log_info += f"Error in calculating {metric_name}: {traceback.format_exc()}\n{e}\n"
        return evaluate_result, log_info

    @staticmethod
    def _get_ratio_output(output, ratio):
        if isinstance(output, dict):
            if ratio in output:
                return output[ratio]
            if "None" in output:
                return output["None"]
            return next(iter(output.values()))
        return output

    @staticmethod
    def _pad_prediction(prediction, target_length, pad_value):
        prediction = np.asarray(prediction).reshape(-1)
        remaining_length = target_length - len(prediction)
        if remaining_length > 0:
            prediction = np.pad(
                prediction,
                (0, remaining_length),
                mode="constant",
                constant_values=pad_value,
            )
        return prediction

    @staticmethod
    def accepted_metrics():
        return tuple(
            list(classification_metrics_score.__all__)
            + [
                metric_name
                for metric_name in classification_metrics_label.__all__
                if metric_name not in classification_metrics_score.__all__
            ]
        )


class UnFixedDetectBoth(AnomalyDetectBoth):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label


class FixedDetectBoth(AnomalyDetectBoth):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label


class AllDetectBoth(AnomalyDetectBoth):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label
