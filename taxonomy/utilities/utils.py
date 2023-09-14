from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np
import pandas as pd
import sklearn
import torch

from taxonomy.utilities.data import Node

if TYPE_CHECKING:
	from taxonomy.utilities.findings import ModelOutputsNode


Array_Series: TypeAlias = Union[np.ndarray, pd.Series]
Array_Series_DataFrame: TypeAlias = Union[np.ndarray, pd.Series, pd.DataFrame]

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


def node_type_checker(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		if 'node' in kwargs:
			node = kwargs['node']
		elif len(args) > 0:
			node, *args = args
		else:
			raise ValueError("node must be passed as a keyword argument or positional argument")

		kwargs['node'] = node.name if isinstance(node, Node) else node
		return func(*args, **kwargs)
	return wrapper


def get_y_yhat(**kwargs) -> tuple[np.ndarray, np.ndarray]:

	try:

		if 'model_outputs_node' in kwargs:
			y    = kwargs['model_outputs_node'].truth
			yhat = kwargs['model_outputs_node'].pred
		else:
			y    = kwargs.get('y')    or kwargs.get('truth')
			yhat = kwargs.get('yhat') or kwargs.get('pred')

		if len(y.shape) > 1:
			raise ValueError("y must be 1-dimensional")

		if isinstance(y, pd.Series):
			y, yhat = y.to_numpy(), yhat.to_numpy()

		return y, yhat

	except KeyError as e:
		" The KeyError would occur if the else clauses are reached and kwargs does not contain the keys 'y' or 'yhat'."
		raise KeyError(f"Key {e} not found in kwargs") from e

	except AttributeError as e:
		"The AttributeErrorwould occur if the model_outputs_node object does not have the attributes truth or pred."
		raise AttributeError(f"Attribute {e} not found in kwargs") from e

	except TypeError as e:
		" A TypeError would occur if you attempt an operation on a variable that is of an inappropriate type. In this case, if model_outputs is a basic data type like an integer, string, etc., that does not support attribute access."
		raise TypeError(f"TypeError: {e}") from e

	except Exception as e:
		raise Exception(f"Exception: {e}") from e # sourcery skip: raise-specific-error


def remove_null_samples(RAISE_ValueError_IF_EMPTY: bool = False, **kwargs) -> Union[dict[str, Array_Series], 'ModelOutputsNode']:
	""" Filter out null samples and check if calculation should proceed. """

	# Identify non-null truth values
	truth = next( (kwargs[key] for key in ['truth', 'y', 'model_outputs'] if key in kwargs), None)
	if truth is None:
		raise ValueError( "truth must be passed as keyword argument named 'truth', 'y', 'model_outputs'")

	non_null = ~np.isnan( truth ) if isinstance( truth, np.ndarray ) else truth.notnull()

	# If needed, check for the existence of at least one non-null value
	if RAISE_ValueError_IF_EMPTY and (not non_null.any()):
		raise ValueError( "y and yhat must contain at least one non-null value" )

	# If ModelOutputs is passed,
	if 'model_outputs' in kwargs:

		# If other arguments are passed along with ModelOutputs, raise ValueError
		if len(kwargs) > 1:
			raise ValueError( "If model_outputs is passed, no other arguments should be passed" )

		# Filter non-null samples from ModelOutputs
		model_outputs: 'ModelOutputsNode' = kwargs['model_outputs']
		model_outputs.truth = model_outputs.truth[non_null]
		model_outputs.loss  = model_outputs.loss[non_null]
		model_outputs.logit = model_outputs.logit[non_null]
		model_outputs.pred  = model_outputs.pred[non_null]
		return model_outputs

	else:
		# If ModelOutputs isn't passed, filter non-null samples from all passed arrays
		for narray in kwargs:
			kwargs[narray] = kwargs[narray][non_null]
		return kwargs


@dataclass
class ROC:
	y: np.ndarray
	yhat: np.ndarray
	REMOVE_NULL: bool = True
	THRESHOLD: float = field( default=None )
	FPR: np.ndarray = field( default=None )
	TPR: np.ndarray = field( default=None )

	def __post_init__(self):
		if self.REMOVE_NULL:
			self.y, self.yhat = remove_null_samples( truth=self.y, yhat=self.yhat, RAISE_ValueError_IF_EMPTY=False )

	def calculate(self) -> 'ROC':
		if self.y.size > 0:
			fpr, tpr, thr = sklearn.metrics.roc_curve( self.y, self.yhat )
			self.FPR, self.TPR, self.THRESHOLD = fpr, tpr, thr[np.argmax( tpr - fpr )]
		return self


@dataclass
class PrecisionRecall:
	y: np.ndarray
	yhat: np.ndarray
	REMOVE_NULL: bool = True
	THRESHOLD: float = field( default=None )
	PPV: np.ndarray = field( default=None )
	RECALL: np.ndarray = field( default=None )
	F_SCORE: np.ndarray = field( default=None )

	def __post_init__(self):
		if self.REMOVE_NULL:
			self.y, self.yhat = remove_null_samples( truth=self.y, yhat=self.yhat, RAISE_ValueError_IF_EMPTY=False )

	def calculate(self) -> 'PrecisionRecall':
		if self.y.size > 0:
			ppv, recall, th = sklearn.metrics.precision_recall_curve( self.y, self.yhat )
			f_score = 2 * (ppv * recall) / (ppv + recall)
			self.THRESHOLD, self.PPV, self.RECALL, self.F_SCORE = th[np.argmax( f_score )], ppv, recall, f_score
		return self

