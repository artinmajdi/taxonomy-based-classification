from __future__ import annotations

import argparse
import enum
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Union, Dict
import pandas as pd

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


# TODO: Need to change the DatasetNames.members() usages to [PC, NIH, CHEXPERT] only
@members
class DatasetNames(enum.Enum):
	PC         = 'PC'
	NIH        = 'NIH'
	CheXPERT   = 'CheX'
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
class MethodNames(enum.Enum):
	BASELINE = 'baseline'
	LOGIT    = 'logit_based'
	LOSS     = 'loss_based'


@members
class ModelNames(enum.Enum):
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


def reading_user_input_arguments(argv=None, jupyter=True, config_name='config.json', **kwargs) -> argparse.Namespace:

	def parse_args() -> argparse.Namespace:
		"""	Getting the arguments from the command line
			Problem:	Jupyter Notebook automatically passes some command-line arguments to the kernel.
						When we run argparse.ArgumentParser.parse_args(), it tries to parse those arguments, which are not recognized by your argument parser.
			Solution:	To avoid this issue, you can modify your get_args() function to accept an optional list of command-line arguments, instead of always using sys.argv.
						When this list is provided, the function will parse the arguments from it instead of the command-line arguments. """

		# If argv is not provided, use sys.argv[1: ] to skip the script name
		args = [] if jupyter else (argv or sys.argv[1:])

		args_list = [
				# Dataset
				dict(name = 'datasetName', type = str, help = 'Name of the dataset'               ),
				dict(name = 'data_mode'   , type = str, help = 'Dataset mode: train or valid'      ),
				dict(name = 'max_sample'  , type = int, help = 'Maximum number of samples to load' ),

				# Model
				dict(name='modelName'   , type=str , help='Name of the pre_trained model.' ),
				dict(name='architecture' , type=str , help='Name of the architecture'       ),

				# Training
				dict(name = 'batch_size'     , type = int   , help = 'Number of batches to process' ),
				dict(name = 'n_epochs'       , type = int   , help = 'Number of epochs to process'  ),
				dict(name = 'learning_rate'  , type = float , help = 'Learning rate'                ),
				dict(name = 'n_augmentation' , type = int   , help = 'Number of augmentations'      ),

				# Hyperparameter Optimization
				dict(name = 'parent_condition_mode', type = str, help = 'Parent condition mode: truth or predicted' ),
				dict(name = 'methodName'             , type = str, help = 'Hyper parameter optimization methodName' ),
				dict(name = 'max_evals'            , type = int, help = 'Number of evaluations for hyper parameter optimization' ),
				dict(name = 'n_batches_to_process' , type = int, help = 'Number of batches to process' ),

				# MLFlow
				dict(name='RUN_MLFLOW'            , type=bool  , help='Run MLFlow'                                             ),
				dict(name='KILL_MLFlow_at_END'    , type=bool  , help='Kill MLFlow'                                            ),

				# Config
				dict(name='config'                , type=str   , help='Path to config file' , DEFAULT='config.json'             ),
				]

		# Initializing the parser
		parser = argparse.ArgumentParser()

		# Adding arguments
		for g in args_list:
			parser.add_argument(f'--{g["name"].replace("_","-")}', type=g['type'], help=g['help'], DEFAULT=g.get('DEFAULT')) # type: ignore

		# Filter out any arguments starting with '-f'
		filtered_argv = [arg for arg in args if not (arg.startswith('-f') or 'jupyter/runtime' in arg.lower())]

		# Parsing the arguments
		return parser.parse_args(args=filtered_argv)

	def updating_config_with_kwargs(updated_args):
		if kwargs and kwargs:
			for key in kwargs:
				updated_args[key] = kwargs[key]
		return updated_args

	def get_config(args):  # type: (argparse.Namespace) -> argparse.Namespace

		# Loading the config.json file
		config_dir = os.path.join(os.path.dirname(__file__), config_name if jupyter else args.config)

		if os.path.exists(config_dir):
			with open(config_dir) as f:
				config_raw = json.load(f)

			# converting args to dictionary
			args_dict = vars(args) if args else {}

			# Updating the config with the arguments as command line input
			updated_args ={key: args_dict.get(key) or values for key, values in config_raw.items() }

			# Updating the config. Used for facilitating the jupyter notebook access
			updated_args = updating_config_with_kwargs(updated_args)

			# Convert the dictionary to a Namespace
			args = argparse.Namespace(**updated_args)

			# Updating the paths to their absolute path
			PATH_BASE = pathlib.Path(__file__).parent.parent.parent.parent
			args.PATH_LOCAL            = PATH_BASE / args.PATH_LOCAL
			args.PATH_DATASETS         = PATH_BASE / args.PATH_DATASETS
			args.PATH_CHEXPERT_WEIGHTS = PATH_BASE / args.PATH_CHEXPERT_WEIGHTS

			args.methodName  = MethodNames[args.methodName.upper()]
			args.datasetName = DatasetNames[args.datasetName.upper()]
			args.modelName   = ModelNames[args.modelName.upper()]

			args.DEFAULT_FINDING_FOLDER_NAME = f'{args.datasetName}-{args.modelName}'

		return args

	# Updating the config file
	return  get_config(args=parse_args())


if __name__ == '__main__':
	print(ExperimentStageNames.members())

	