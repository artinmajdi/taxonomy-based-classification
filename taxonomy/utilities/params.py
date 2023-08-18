from __future__ import annotations

import argparse
import enum
from dataclasses import dataclass
from typing import Union, Dict

import pandas as pd

from taxonomy.utilities.data import Data


def members(cls):
	
	# Add the members class method
	cls.members = classmethod(lambda cls2: list(cls2.__members__))

	# cls.values = classmethod(lambda cls2: [n.value for n in cls2.__members__])
	
	# Make the class iterable
	cls.__iter__ = lambda self: iter(self.__members__.keys())
	
	# Overwrite the __str__ method, to output only the name of the member
	cls.__str__ = lambda self: self.value
	return cls


@members
class ExperimentStageNames(enum.Enum):
	ORIGINAL = 'original'
	NEW      = 'updated'


@members
class DatasetNames(enum.Enum):
	PC   = 'PC'
	NIH  = 'NIH'
	CheX = 'CheX'
	
	
@members
class ThreshTechList(enum.Enum):
	DEFAULT          = 'DEFAULT'
	ROC              = 'ROC'
	PRECISION_RECALL = 'PRECISION_RECALL'
	
	
@members
class DataModes(enum.Enum):
	TRAIN = 'train'
	TEST  = 'test'
	
	
@members
class MethodNames(enum.Enum):
	BASELINE = 'baseline'
	LOGIT    = 'logit_based'
	LOSS     = 'loss_based'


@members
class ModelNames(enum.Enum):
	NIH                   = 'densenet121-res224-nih'
	RSNA                  = 'densenet121-res224-rsna'
	PC                    = 'densenet121-res224-pc'
	CHEXPERT              = 'densenet121-res224-chex'
	MIMIC_NB              = 'densenet121-res224-mimic_nb'
	MIMIC_CH              = 'densenet121-res224-mimic_ch'
	ALL_224               = 'densenet121-res224-all'
	ALL_512               = 'resnet50-res512-all'
	BASELINE_JFHEALTHCARE = 'baseline_jfhealthcare'
	BASELINE_CHEX         = 'baseline_CheX'


@members
class EvaluationMetricNames(enum.Enum):
	ACC = 'acc'
	AUC = 'auc'
	F1  = 'f1'
	

@members
class ModelFindingNames(enum.Enum):
	GROUND_TRUTH = 'truth'
	LOSS_VALUES  = 'loss'
	LOGIT_VALUES = 'logit'
	PRED_PROBS   = 'pred'
	
	
@dataclass
class NodeData:
	hierarchy_penalty: pd.DataFrame
	metrics          : Union[Dict, pd.DataFrame]
	data             : Union[pd.DataFrame, Dict[str, pd.DataFrame]]



if __name__ == '__main__':
	print(ExperimentStageNames.members())

	