import argparse
import json
import pathlib
import sys

import pydantic
from pydantic import validator

from taxonomy.utilities.params import DataModes, DatasetNames, EvaluationMetricNames, LossFunctionOptions, \
	ModelWeightNames, \
	ParentMetricToUseNames, SimulationOptions, TechniqueNames


class DatasetSettings(pydantic.BaseModel):
	datasetNames: list[DatasetNames] = pydantic.Field(default_factory = lambda: [DatasetNames.PC, DatasetNames.NIH, DatasetNames.CHEXPERT], description='list of dataset names. e.g. ["PC", "NIH", "CheX"]')
	views: list[str] = pydantic.Field(default_factory = lambda : ['PA', 'AP'], description='list of views. e.g. ["PA", "AP"]')
	non_null_samples: bool         = True
	path            : pathlib.Path = pathlib.Path('./datasets')
	data_mode       : DataModes    = DataModes.TRAIN
	max_samples     : pydantic.conint(gt=0)         = 1000
	train_test_ratio: pydantic.confloat(ge=0, le=1) = 0.7

class TrainingSettings(pydantic.BaseModel):
	criterion	      : LossFunctionOptions = LossFunctionOptions.BCE
	batches_to_process: pydantic.conint(ge   = 1) = 10
	augmentation_count: pydantic.conint(ge   = 0) = 1
	learning_rate     : pydantic.confloat(gt = 0) = 0.0001
	batch_size        : pydantic.conint(ge   = 1) = 1000
	epochs            : pydantic.conint(ge   = 1) = 3
	shuffle: bool = False
	silent : bool = True

class ModelSettings(pydantic.BaseModel):
	name           : ModelWeightNames = ModelWeightNames.ALL_224
	chexpert_weights_path: pathlib.Path     = pathlib.Path("./pre_trained_models/chestxray/chexpert_baseline_model_weight.zip")

	@property
	def full_name(self) -> str:
		return self.name.full_name



class SimulationSettings(pydantic.BaseModel):
	findings_original: SimulationOptions = SimulationOptions.RUN_SIMULATION
	findings_new     : SimulationOptions = SimulationOptions.RUN_SIMULATION
	hyperparameters  : SimulationOptions = SimulationOptions.RUN_SIMULATION
	metrics          : SimulationOptions = SimulationOptions.RUN_SIMULATION

class HyperParameterTuningSettings(pydantic.BaseModel):
	metric_used_to_select_best_parameters: EvaluationMetricNames = EvaluationMetricNames.AUC
	techniqueName         : TechniqueNames         = TechniqueNames.LOGIT
	parent_metric_to_use  : ParentMetricToUseNames = ParentMetricToUseNames.TRUTH
	max_evals             : pydantic.conint(gt = 0) = 20
	num_workers           : pydantic.conint(gt = 0) = 1
	hyper_param_multiplier: pydantic.confloat(ge = 0) = 0.0
	hyper_param_additive  : pydantic.confloat(ge = 0) = 1.0
	use_parallelization   : bool = True

class OutputSettings(pydantic.BaseModel):
	path: pathlib.Path = pathlib.Path('../outputs')

	@validator('path', pre=False, always=True)
	def make_path_absolute(cls, v):
		return v.resolve()

class Settings(pydantic.BaseSettings):

	dataset                    : DatasetSettings
	training                   : TrainingSettings
	model                      : ModelSettings
	simulation                 : SimulationSettings
	hyperparameter_tuning      : HyperParameterTuningSettings
	output                     : OutputSettings

	class Config:
		use_enum_values      = False
		case_sensitive       = False
		str_strip_whitespace = True


def get_settings(argv=None, jupyter=True, config_name='config.json') -> Settings | ValueError:

	def parse_args() -> dict:
		"""	Getting the arguments from the command line
			Problem:	Jupyter Notebook automatically passes some command-line arguments to the kernel.
						When we run argparse.ArgumentParser.parse_args(), it tries to parse those arguments, which are not recognized by your argument parser.
			Solution:	To avoid this issue, you can modify your get_args() function to accept an optional list of command-line arguments, instead of always using sys.argv.
						When this list is provided, the function will parse the arguments from it instead of the command-line arguments. """

		# If argv is not provided, use sys.argv[1: ] to skip the script name
		args = [] if jupyter else (argv or sys.argv[1:])

		args_list = [
				# Dataset
				dict(name = 'datasetNames', type = str, help = 'Name of the dataset'               ),
				dict(name = 'data_mode'   , type = str, help = 'Dataset mode: train or valid'      ),
				dict(name = 'max_samples' , type = int, help = 'Maximum number of samples to load' ),

				# Model
				dict(name='techniqueName'   , type=str , help='Name of the pre_trained model.' ),

				# Training
				dict(name = 'batch_size'        , type = int  , help = 'Number of batches to process' ),
				dict(name = 'epochs'            , type = int  , help = 'Number of epochs to process'  ),
				dict(name = 'learning_rate'     , type = float, help = 'Learning rate'                ),
				dict(name = 'augmentation_count', type = int  , help = 'Number of augmentations'      ),

				# Hyperparameter Optimization
				dict(name = 'parent_metric_to_use', type = str, help = 'Parent condition mode: truth or predicted' ),
				dict(name = 'max_evals'           , type = int, help = 'Number of evaluations for hyper parameter optimization' ),
				dict(name = 'batches_to_process'  , type = int, help = 'Number of batches to process' ),

				# Config
				dict(name='config'                , type=str   , help='Path to config file'),
				]

		# Initializing the parser
		parser = argparse.ArgumentParser()

		# Adding arguments
		for g in args_list:
			parser.add_argument(f'--{g["name"].replace("_","-")}', type=g['type'], help=g['help'], default=g.get('default')) # type: ignore

		# Filter out any arguments starting with '-f'
		filtered_argv = [arg for arg in args if not (arg.startswith('-f') or 'jupyter/runtime' in arg.lower())]

		# Parsing the arguments
		parsed_args = parser.parse_args(args=filtered_argv)

		return {k: v for k, v in vars(parsed_args).items() if v is not None}

	def get_config(args_dict: argparse.Namespace) -> Settings | ValueError:

		# Loading the config.json file
		config_dir = pathlib.Path(args_dict.get('config') or config_name).resolve()

		if not config_dir.exists():
			return ValueError(f'Config file not found at {config_dir}')

		with open(config_dir) as f:
			config_data = json.load(f)

		# Updating the config with the arguments as command line input
		def  update_config(model, config_key):
			for key in model.__fields__:
				if key in args_dict:
					config_data[config_key][key] = args_dict[key]

		# Updating the config with the arguments as command line input
		update_config(DatasetSettings             , 'dataset')
		update_config(TrainingSettings            , 'training')
		update_config(ModelSettings               , 'model')
		update_config(SimulationSettings          , 'simulation')
		update_config(HyperParameterTuningSettings, 'hyperparameter_tuning')
		update_config(OutputSettings              , 'output')

		# Convert the dictionary to a Namespace
		config = Settings(**config_data)

		# # Updating the paths to their absolute path
		# PATH_BASE = pathlib.Path(__file__).parent.parent.parent.parent
		# args.DEFAULT_FINDING_FOLDER_NAME = f'{args.datasetName}-{args.modelName}'

		return config

	# Updating the config file
	return  get_config(args_dict=parse_args())