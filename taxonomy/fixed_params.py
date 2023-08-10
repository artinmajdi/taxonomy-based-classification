import enum

def members(cls):
	# Add the members class method
	cls.members = classmethod(lambda cls: list(cls.__members__))
	# Make the class iterable
	cls.__iter__ = lambda self: iter(self.__members__.keys())
	# Overwrite the __str__ method, to output only the name of the member
	cls.__str__ = lambda self: self.value
	return cls


@members
class ExperimentSTAGE(enum.Enum):
	ORIGINAL = 'ORIGINAL'
	NEW      = 'NEW'

@members
class DatasetList(enum.Enum):
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
	LOGIT    = 'on_logit'
	LOSS     = 'on_loss'