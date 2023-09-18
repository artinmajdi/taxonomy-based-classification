import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field, InitVar
from typing import Any, TypeAlias, Union

from pydantic import BaseModel, confloat, conint, Field, FieldValidationInfo
from pydantic.functional_validators import field_validator

from taxonomy.utilities.params import DataModes, DatasetNames, EvaluationMetricNames, LossFunctionOptions, ModelWeightNames, ParentMetricToUseNames, SimulationOptions, TechniqueNames, ThreshTechList

PathNoneType: TypeAlias = Union[pathlib.Path, None]

@dataclass
class DatasetInfo:
	datasetName      : DatasetNames
	path_all_datasets: pathlib.Path
	views	         : list[str]
	USE_INCLUDED_FILE_IN_TORCHXRAYVISION: bool = True
	data_mode        : InitVar[DataModes] = field(default=None)
	path             : pathlib.Path  	  = field(default=None)
	csv_path         : PathNoneType		  = field(default=None)
	metadata_path    : PathNoneType       = field(default=None)
	params_config    : dict[str, Any]     = field(default=None)

	def __post_init__(self, data_mode: DataModes):

		self.path = self._get_dataset_relative_path(data_mode)
		self.params_config = {'imgpath':self.path, 'views':self.views}

		if self.datasetName == DatasetNames.MIMIC:
			self.metadata_path = self.resolve_path('MIMIC/mimic-cxr-2.0.0-metadata.csv.gz')
			self.csv_path 	   = self.resolve_path('MIMIC/mimic-cxr-2.0.0-chexpert.csv.gz')

			self.params_config['csvpath']     = self.csv_path,
			self.params_config['metacsvpath'] = self.metadata_path

		else:

			# Returning USE_INCLUDED_FILE if USE_INCLUDED_FILE_IN_TORCHXRAYVISION is True else False
			from torchxrayvision.datasets import USE_INCLUDED_FILE
			doUFI = self.USE_INCLUDED_FILE_IN_TORCHXRAYVISION and USE_INCLUDED_FILE

			# Updating the bbox list path
			if self.datasetName == DatasetNames.NIH:
				self.params_config['bbox_list_path'] = doUFI or self.resolve_path('NIH/BBox_List_2017.csv')

			# Updating the csv path
			self.csv_path = doUFI or self._get_relative_csv_path(data_mode)

			self.params_config['csvpath'] = self.csv_path


	def resolve_path(self, base_path: Union[str, None]) -> Union[pathlib.Path, None]:
		return base_path and self.path_all_datasets / pathlib.Path(base_path)

	def _get_relative_csv_path(self, data_mode: DataModes) -> pathlib.Path:
		csv_path_dict = {
				'nih'     : 'NIH/Data_Entry_2017.csv',
				'rsna'    : 'RSNA/stage_2_train_labels,csv',
				'pc'      : 'PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',
				'chex'    : 'CheX/CheXpert-v1.0-small/{data_mode.value}.csv',
				'vinbrain': f'VinBrain/dicom/{data_mode.value}.csv',
				}
		return self.resolve_path( csv_path_dict.get(self.datasetName.value) )


	def _get_dataset_relative_path(self, data_mode: DataModes) -> pathlib.Path:
		dataset_path_dict = {
				'nih'     : 'NIH/images-224',
				'rsna'    : 'RSNA/images-224',
				'pc'      : 'PC/images-224',
				'chex'    : 'CheX/CheXpert-v1.0-small',
				'openai'  : 'Openi/NLMCXR_png',
				'vinbrain': f'VinBrain/{data_mode.value}',
				'mimic'   : 'MIMIC/re_512_3ch'
				}
		return self.resolve_path( dataset_path_dict.get(self.datasetName.value) )



class DatasetSettings(BaseModel):
	data_mode        : DataModes          = DataModes.TRAIN
	views            : list[str]          = Field(default_factory = lambda: ['PA', 'AP'])
	path_all_datasets: pathlib.Path       = pathlib.Path('./datasets')
	datasetNames     : list[DatasetNames] = Field(default_factory = lambda: [DatasetNames.PC, DatasetNames.NIH, DatasetNames.CHEXPERT])
	datasetInfoList  : list[DatasetInfo]   = Field(default = None)
	non_null_samples : bool                = True
	max_samples      : conint(gt=0)        = 1000
	train_test_ratio : confloat(ge=0,le=1) = 0.7
	default_taxonomy : dict[str, set[str]]

	@field_validator('path_all_datasets', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return v.resolve()

	@field_validator('datasetInfoList', mode='before')
	def post_process_info(cls, v: None, info: FieldValidationInfo) -> list[DatasetInfo]:
		return [ DatasetInfo(   path_all_datasets = info.data['path_all_datasets'],
								data_mode         = info.data['data_mode'],
								views 			  = info.data['views'],
								datasetName       = dt )
				 for dt in info.data['datasetNames']]


class TrainingSettings(BaseModel):
	criterion	      : LossFunctionOptions = LossFunctionOptions.BCE
	batches_to_process: conint(ge   = 1) = 10
	augmentation_count: conint(ge   = 0) = 1
	learning_rate     : confloat(gt = 0) = 0.0001
	batch_size        : conint(ge   = 1) = 1000
	epochs            : conint(ge   = 1) = 3
	shuffle: bool = False
	silent : bool = True


class ModelSettings(BaseModel):
	modelName            : ModelWeightNames = ModelWeightNames.ALL_224
	chexpert_weights_path: pathlib.Path     = pathlib.Path("./pre_trained_models/chestxray/chexpert_baseline_model_weight.zip")

	@property
	def full_name(self) -> str:
		return self.modelName.full_name


class SimulationSettings(BaseModel):
	findings_original  : SimulationOptions = SimulationOptions.RUN_SIMULATION
	findings_new       : SimulationOptions = SimulationOptions.RUN_SIMULATION
	hyperparameters    : SimulationOptions = SimulationOptions.RUN_SIMULATION
	metrics            : SimulationOptions = SimulationOptions.RUN_SIMULATION
	use_parallelization: bool = True
	num_workers        : conint(gt         = 0) = 1


class TechniqueSettings(BaseModel):
	technique_name                       : TechniqueNames = TechniqueNames.LOGIT
	metric_used_to_select_best_parameters: EvaluationMetricNames = EvaluationMetricNames.AUC
	parent_metric_to_use                 : ParentMetricToUseNames = ParentMetricToUseNames.TRUTH
	threshold_technique                     : ThreshTechList = ThreshTechList.ROC


class HyperParameterTuningSettings(BaseModel):
	max_evals              : conint(gt = 0) = 20
	initial_multiplier     : confloat(ge = 0) = 0.0
	initial_additive       : confloat(ge = 0) = 1.0
	search_space_multiplier: list[float] = [-1, 1]
	search_space_additive  : list[float] = [-4, 4]
	optimization_metric: EvaluationMetricNames = EvaluationMetricNames.AUC


class OutputSettings(BaseModel):
	path: pathlib.Path = pathlib.Path('../outputs')

	@field_validator('path', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return v.resolve()


class Settings(BaseModel):

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


def get_settings(argv=None, jupyter=True, config_path='config.json') -> 'Settings':

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
				dict(name='technique_name'   , type=str , help='Name of the pre_trained model.' ),

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

	def get_config(args_dict: dict) -> Union[Settings,ValueError]:

		# Loading the config.json file
		config_dir = pathlib.Path( args_dict.get('config') or config_path ).resolve()

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

		config.dataset.path_all_datasets = pathlib.Path(config.dataset.path_all_datasets).resolve()
		# # Updating the paths to their absolute path
		# PATH_BASE = pathlib.Path(__file__).parent.parent.parent.parent
		# args.DEFAULT_FINDING_FOLDER_NAME = f'{args.datasetName}-{args.modelName}'

		return config

	# Updating the config file
	return  get_config(args_dict=parse_args())


def main():
	config = get_settings()
	print(config.dataset.datasetInfoList)
	print('sometjhong')


if __name__ == '__main__':
	main()
