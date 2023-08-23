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
	NEW      = 'new'


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
	PC                    = ("pc"           , 'densenet121-res224-pc')
	NIH                   = ("nih"          , 'densenet121-res224-nih')
	CHEXPERT              = ("chex"         , 'densenet121-res224-chex')
	RSNA                  = ("rsna"         , 'densenet121-res224-rsna')
	MIMIC_NB              = ("mimic_nb"     , 'densenet121-res224-mimic_nb')
	MIMIC_CH              = ("mimic_ch"     , 'densenet121-res224-mimic_ch')
	ALL_224               = ("all_224"      , 'densenet121-res224-all')
	ALL_512               = ("all_512"      , 'resnet50-res512-all')
	BASELINE_JFHEALTHCARE = ("baseline_jf"  , 'baseline_jfhealthcare')
	BASELINE_CHEX         = ("baseline_chex", 'baseline_CheX')

	def __new__(cls, short_name, full_name):
		obj = object.__new__(cls)
		obj._value_ = short_name
		obj.full_name = full_name
		return obj


@members
class EvaluationMetricNames(enum.Enum):
	ACC = 'acc'
	AUC = 'auc'
	F1  = 'f1'
	THRESHOLD = 'threshold'


@members
class FindingNames(enum.Enum):
	TRUTH = 'truth'
	LOSS  = 'loss_values'
	LOGIT = 'logit_values'
	PRED  = 'pred_probs'


@members
class HyperparameterNames(enum.Enum):
	MULTIPLIER = 'hyper_param_multiplier'
	ADDITIVE   = 'hyper_param_additive'


@members
class ParentMetricToUseNames( enum.Enum ):
	TRUTH: str = 'truth'
	PRED : str = 'pred_probs'


@members
class SimulationOptions(enum.Enum):
	LOAD_FROM_LOCAL = 'load_from_local'
	RUN_SIMULATION  = 'run_simulation'

