from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Union

import numpy as np
import pandas as pd
import sklearn

from taxonomy.utilities.data import Node
from taxonomy.utilities.params import ExperimentStageNames, ThreshTechList
from taxonomy.utilities.settings import Settings
from taxonomy.utilities.utils import get_y_yhat, node_type_checker, PrecisionRecall, ROC


@dataclass
class CalculateMetricsNode:
	y     : np.ndarray
	yhat  : np.ndarray
	THRESHOLD: float

	def __new__(cls, node: Node, **kwargs) -> 'CalculateMetricsNode':

		kwargs_keys = set(kwargs.keys())

		def get_threshold() -> float:

			if 'THRESHOLD' in kwargs:
				return kwargs['THRESHOLD']

			assert kwargs_keys.intersection({'config', 'threshold_technique'}), "Either config or threshold_technique must be passed"

			th_teq = kwargs.get("threshold_technique") or kwargs.get("config").hyperparameter_tuning.threshold_technique
			return cls.calculate_optimum_threshold(y=y, yhat=yhat, threshold_technique=th_teq)

		y, yhat = get_y_yhat(**kwargs)
		y, yhat = y[node.non_null_indices], yhat[node.non_null_indices]

		THRESHOLD = get_threshold()

		return super().__new__(cls, y=y, yhat=yhat, THRESHOLD=THRESHOLD)

	@staticmethod
	def calculate_optimum_threshold(y: np.ndarray, yhat: np.ndarray, threshold_technique: ThreshTechList) -> float:

		if threshold_technique == ThreshTechList.DEFAULT:
			return 0.5

		if threshold_technique == ThreshTechList.ROC:
			return ROC( y=y, yhat=yhat, REMOVE_NULL=False ).calculate().THRESHOLD

		if threshold_technique == ThreshTechList.PRECISION_RECALL:
			return PrecisionRecall( y=y, yhat=yhat, REMOVE_NULL=False ).calculate().THRESHOLD

		raise NotImplementedError( f"ThreshTechList {threshold_technique} not implemented" )

	@cached_property
	def AUC(self) -> float:
		return sklearn.metrics.roc_auc_score( self.y, self.yhat )

	@cached_property
	def ACC(self) -> float:
		return sklearn.metrics.accuracy_score( self.y, self.yhat > self.THRESHOLD )

	@cached_property
	def F1(self) -> float:
		return sklearn.metrics.f1_score( self.y, self.yhat > self.THRESHOLD )

	@cached_property
	def get_all(self) -> MetricsNode:
		return MetricsNode( ACC=self.ACC, AUC=self.AUC, F1=self.F1, THRESHOLD=self.THRESHOLD )


@dataclass
class MetricsNode:
	AUC      : float = None
	ACC      : float = None
	F1       : float = None
	THRESHOLD: float = None

	@classmethod
	def calculate(cls, y: Union[np.ndarray, pd.Series], yhat: Union[np.ndarray, pd.Series], node: Node, config: Settings) -> 'MetricsNode':
		return CalculateMetricsNode( y=y, yhat=yhat, config=config, node=node).get_all


@dataclass
class Metrics:
	ACC      : pd.Series = None
	AUC      : pd.Series = None
	F1       : pd.Series = None
	THRESHOLD: pd.Series = None

	@classmethod
	def initialize(cls, classes: Optional[pd.Index] = None) -> 'Metrics':
		return cls( ACC = pd.Series( index=classes ),
					AUC = pd.Series( index=classes ),
					F1  = pd.Series( index=classes ),
					THRESHOLD = pd.Series( index=classes ) )

	def save(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'Metrics':

		def get_formatted(metric):
			return metric if isinstance(metric, pd.Series) else pd.Series([metric])

		output_path = config.output.path / f'{experiment_stage}/metrics.xlsx'
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with pd.ExcelWriter( output_path ) as writer:
			get_formatted(self.ACC).to_excel( writer, sheet_name = 'acc' )
			get_formatted(self.AUC).to_excel( writer, sheet_name = 'auc' )
			get_formatted(self.F1).to_excel(  writer, sheet_name = 'f1' )
			get_formatted(self.THRESHOLD).to_excel( writer, sheet_name = 'threshold' )
		return self

	def load(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'Metrics':

		def set_formatted(metric):
			if isinstance(metric, pd.Series) and len(metric) == 1:
				return metric.iloc[0]
			return metric

		input_path = config.output.path / f'{experiment_stage}/metrics.xlsx'

		if input_path.is_file():
			self.ACC = set_formatted(pd.read_excel(input_path, sheet_name = 'acc', index_col = 0, squeeze = True))
			self.AUC = set_formatted(pd.read_excel(input_path, sheet_name = 'auc', index_col = 0, squeeze = True))
			self.F1  = set_formatted(pd.read_excel(input_path, sheet_name = 'f1' , index_col = 0, squeeze = True))
			self.THRESHOLD = set_formatted(pd.read_excel(input_path, sheet_name = 'threshold', index_col = 0, squeeze = True))
		return self

	@node_type_checker
	def __getitem__(self, node: Union[Node, str]) -> 'MetricsNode':
		return MetricsNode( AUC=self.AUC[node], ACC=self.ACC[node], F1=self.F1[node], THRESHOLD=self.THRESHOLD[node] )

	@node_type_checker
	def __setitem__(self, node: Union[Node, str], value: MetricsNode):

		assert isinstance( value, MetricsNode ), "Value must be of type MetricsNode"

		self.AUC[node]       = value.AUC
		self.ACC[node]       = value.ACC
		self.F1[node]        = value.F1
		self.THRESHOLD[node] = value.THRESHOLD

	@classmethod
	def calculate(cls, config: Settings, **kwargs) -> 'Metrics':

		if not set(kwargs.keys()).intersection({'truth', 'pred', 'model_outputs'}):
			raise ValueError("Either truth, pred or model_outputs must be passed")

		truth = kwargs.get('truth') or kwargs['model_outputs'].truth
		pred  = kwargs.get('pred' ) or kwargs['model_outputs'].pred

		classes = truth.columns
		metrics = cls.initialize(classes)

		for node in classes:
			metrics[node] = MetricsNode.calculate( y=truth[node], yhat=pred[node], config=config, node=node )

		return metrics


