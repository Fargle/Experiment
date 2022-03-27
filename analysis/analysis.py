import argparse
from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import eli5
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from eli5 import sklearn
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
from scikeras.wrappers import KerasRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from tensorflow import keras

import analysis_app

ModelAnalyisFunction = Callable[[KerasRegressor, pd.DataFrame, pd.DataFrame], object]
FeatureAnalysisFunction = Callable[[pd.DataFrame, pd.DataFrame], object]


@dataclass
class AnalysisAssets:
    """Analysis Assets"""

    model: KerasRegressor
    X: pd.DataFrame
    y: pd.DataFrame
    features: List[str]


@dataclass
class FeatureAnalysis:
    "Feature Analysis"

    analyze: FeatureAnalysisFunction
    assets: AnalysisAssets

    def conduct(self):
        """Conduct Feature analysis"""
        return self.analyze(self.assets.X, self.assets.y)


def mutual_info_scores(X: pd.DataFrame, y: pd.DataFrame):
    """Do Mutual information on features X with respect to indpendent y"""
    mi_scores = mutual_info_regression(X, y)
    return mi_scores


@dataclass
class ModelAnalysis:
    """Model Analysis"""

    analyze: ModelAnalyisFunction
    assets: AnalysisAssets

    def conduct(self):
        """Execute model analysis"""
        return self.analyze(self.assets.model, self.assets.X, self.assets.y)


def permutation_importance(
    model: KerasRegressor, test_X: pd.DataFrame, test_y: pd.DataFrame
):
    """Do permutation importance on data"""
    perm = PermutationImportance(model, random_state=1)
    perm.fit(test_X[0:1000], test_y[0:1000])
    return perm


def pdp_plot_closure(feature, feature_names):
    """Closure"""

    def pdp_plot(model: KerasRegressor, test_X: pd.DataFrame, test_y: pd.DataFrame):
        """Partial Dependency Plot"""
        pdp_goals = pdp.pdp_isolate(
            model=model,
            dataset=test_X,
            model_features=feature_names,
            feature=feature,
        )
        return pdp.pdp_plot(pdp_goals, feature)

    return pdp_plot


def r_squared(model, test_X, test_y):
    """Get R squared"""
    predictions = model.predict(test_X)
    return r2_score(test_y, predictions)


def confusion_matrix():
    pass


def principal_components():
    pass


def mutual_information():
    pass


def load_train_test(path: str):
    """Load the train/test data"""
    data_to_gather = ["train_X", "test_X", "train_y", "test_y"]
    data = {data: pd.read_csv(f"{path}/{data}.csv") for data in data_to_gather}
    return data


def load_model(model_path):
    """Load model"""
    return keras.models.load_model(model_path)


def sklearn_wrapped(model):
    """Wraps model in sklearn implementation"""
    regressor = KerasRegressor(model)
    return regressor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze your experiments")
    parser.add_argument("-experiment", "-exp", type=str, help="Path to experiment.")
    return parser.parse_args()


def run_app(app):
    """Run Front End App"""
    print("Inside Run_app")
    app.build()
    app.run()


def main():
    """Main"""
    print("Inside Main")
    arguments = parse_args()

    # load data
    data = load_train_test(f"{arguments.experiment}")

    # load model
    model_path = f"{arguments.experiment}/model"
    model = load_model(model_path)
    model = sklearn_wrapped(model)
    model.initialize(data["test_X"], data["test_y"])

    analysis_assets = AnalysisAssets(
        model=model,
        X=data["test_X"],
        y=data["test_y"],
        features=list(data["test_X"].columns),
    )

    model_analysis = {
        "Permutation Importance": ModelAnalysis(
            permutation_importance, analysis_assets
        ),
        "PDP": ModelAnalysis(
            pdp_plot_closure(lambda: feature, lambda: get_features), analysis_assets
        ),
    }
    f_analysis = FeatureAnalysis(mutual_info_scores, analysis_assets)

    callbacks = {
        "Feature Analysis": f_analysis,
        "Model Analysis": model_analysis,
    }

    app = analysis_app.App(callbacks)
    run_app(app)


if __name__ == "__main__":
    main()
