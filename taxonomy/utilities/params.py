from __future__ import annotations

import enum


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
	PC         = 'PC'
	NIH        = 'NIH'
	CHEXPERT   = 'CheX'
	RSNA       = 'RSNA'
	MIMIC      = 'MIMIC'
	ALL        = 'ALL'
	VINBRAIN   = 'VinBrain'
	OPENI      = 'Openi'
	NIH_GOOGLE = 'NIH_Google'
	
	
@members
class ThreshTechList(enum.Enum):
	DEFAULT          = 'DEFAULT'
	ROC              = 'ROC'
	PRECISION_RECALL = 'PRECISION_RECALL'


@members
class DataModes(enum.Enum):
	TRAIN = 'train'
	TEST  = 'test'
	ALL   = 'all'


@members
class TechniqueNames(enum.Enum):
	BASELINE = 'baseline'
	LOGIT    = 'logit_based'
	LOSS     = 'loss_based'


@members
class ModelWeightNames(enum.Enum):
	PC                    = 'densenet121-res224-pc'
	NIH                   = 'densenet121-res224-nih'
	CHEXPERT              = 'densenet121-res224-chex'
	RSNA                  = 'densenet121-res224-rsna'
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
	THRESHOLD = 'threshold'


@members
class FindingNames(enum.Enum):
	GROUND_TRUTH = 'ground_truth'
	LOSS_VALUES  = 'loss_values'
	LOGIT_VALUES = 'logit_values'
	PRED_PROBS   = 'pred_probs'


@members
class HyperparameterNames(enum.Enum):
	A = 'a'
	B = 'b'


@members
class ParentMetricToUseNames( enum.Enum ):
	TRUTH     : str = 'truth'
	PREDICTED : str = 'predicted'


@members
class SimulationOptions(enum.Enum):
	LOAD_LOCAL_FINDINGS = 'load_local_findings'
	LOAD_FROM_MLFLOW	= 'load_from_mlflow'
	RUN_SIMULATION		= 'run_simulation'

