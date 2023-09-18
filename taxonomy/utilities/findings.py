from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from matplotlib import pyplot as plt

from taxonomy.utilities.data import LoadSaveFile
from taxonomy.utilities.metrics import Metrics
from taxonomy.utilities.params import EvaluationMetricNames, ExperimentStageNames, TechniqueNames
from taxonomy.utilities.utils import node_type_checker

if TYPE_CHECKING:
	from taxonomy.utilities.settings import Settings
	from taxonomy.utilities.model import ModelType
	from taxonomy.utilities.hyperparameters import HyperParameters, HyperPrametersNode
	from taxonomy.utilities.data import Node, Data



USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


@dataclass
class ModelOutputsNode:
	truth: Union[np.ndarray, pd.Series] = None
	loss : Union[np.ndarray, pd.Series] = None
	logit: Union[np.ndarray, pd.Series] = None
	pred : Union[np.ndarray, pd.Series] = None

	def to_numpy(self) -> 'ModelOutputsNode':
		for key in ['truth', 'loss', 'logit', 'pred']:
			value = getattr(self, key)
			if isinstance(value, pd.Series):
				setattr(self, key, value.to_numpy())
		return self

	def to_series(self, classes: list[str]) -> 'ModelOutputsNode':
		for key in ['truth', 'loss', 'logit', 'pred']:
			value = getattr(self, key)
			if isinstance(value, np.ndarray):
				setattr(self, key, pd.Series(value, index=classes))
		return self


@dataclass
class ModelOutputs:
	""" Class for storing model-related findings. """
	truth: pd.DataFrame = None
	loss : pd.DataFrame = None
	logit: pd.DataFrame = None
	pred : pd.DataFrame = None


	@classmethod
	def initialize(cls, index: Optional[pd.Index] = None, columns: Optional[pd.Index] = None) -> 'ModelOutputs':
		return cls( truth = pd.DataFrame( columns=columns, index=index ),
					loss  = pd.DataFrame( columns=columns, index=index ),
					logit = pd.DataFrame( columns=columns, index=index ),
					pred  = pd.DataFrame( columns=columns, index=index ))


	def save(self, config: 'Settings', experiment_stage: ExperimentStageNames) -> 'ModelOutputs':

		output_path = config.output.path / f'{experiment_stage}/model_outputs.xlsx'
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with pd.ExcelWriter( output_path ) as writer:
			self.truth.to_excel( writer, sheet_name ='truth' )
			self.loss.to_excel( writer, sheet_name ='loss' )
			self.logit.to_excel( writer, sheet_name = 'logit' )
			self.pred.to_excel( writer , sheet_name = 'pred' )
		return self


	def load(self, config: 'Settings', experiment_stage: ExperimentStageNames) -> 'ModelOutputs':

		input_path = config.output.path / f'{experiment_stage}/model_outputs.xlsx'

		if input_path.is_file():
			self.truth = pd.read_excel( input_path, sheet_name  ='truth' )
			self.loss  = pd.read_excel( input_path, sheet_name  ='loss' )
			self.logit = pd.read_excel( input_path, sheet_name  = 'logit')
			self.pred  = pd.read_excel( input_path, sheet_name  = 'pred' )
		return self


	@node_type_checker
	def __getitem__(self, node: Union[Node, str]) -> 'ModelOutputsNode':
		return ModelOutputsNode( truth=self.truth[node], loss=self.loss[node],
								 logit=self.logit[node], pred=self.pred[node] )


	@node_type_checker
	def __setitem__(self, node: Union[Node, str], value: ModelOutputsNode):

		assert isinstance( value, ModelOutputsNode ), "Value must be of type ModelOutputsNode"

		self.truth[node] = value.truth
		self.loss[node]  = value.loss
		self.logit[node] = value.logit
		self.pred[node]  = value.pred


@dataclass
class FindingsNode:
	config              : 'Settings'
	node                : Node
	hyperparameters_node: HyperPrametersNode = None
	model_outputs_node  : ModelOutputsNode   = None


@dataclass
class Findings:
	""" Class for storing overall findings including configuration, data, and metrics. """
	config          : 'Settings'
	data            : 'Data' = None
	model           : ModelType = None
	model_outputs   : ModelOutputs = field(default = None, init = False)
	metrics         : Metrics = field(default = None, init = False)
	hyperparameters : HyperParameters = None
	experiment_stage: ExperimentStageNames = ExperimentStageNames.ORIGINAL

	def update_metrics(self):
		self.metrics = Metrics.calculate( config=self.config, model_outputs=self.model_outputs )

@dataclass
class FindingsTrainTest:
	TRAIN: Findings = field( default = None )
	TEST : Findings = field( default = None )


@dataclass
class FindingsAllTechniques:
	LOSS    : FindingsTrainTest = field( default = None )
	LOGIT   : FindingsTrainTest = field( default = None )
	BASELINE: FindingsTrainTest = field( default = None )

	def _get_obj(self):
		for objName in ['loss', 'logit', 'baseline']:
			obj = getattr( self, objName )
			if obj is not None:
				return obj
		raise ValueError(
			'No suitable object has been initialized. Please initialize either "baseline", "loss", or "logit".' )

	@property
	def nodes(self):
		return self._get_obj().TEST.data.nodes

	@property
	def config(self):
		return self._get_obj().TEST.config


	def plot_roc_curves(self, save_figure=True, figsize=(15, 15), font_scale=1.8, fontsize=20, labelpad=0):

		impacted_nodes = self.nodes.IMPACTED
		list_parent_nodes = list( self.nodes.taxonomy.keys() )
		truth = self.BASELINE.TEST.model_outputs.truth

		# Set up the grid
		def setup_plot():
			nonlocal fig, axes, n_rows, n_cols

			# Set a seaborn style for visually appealing plots
			sns.set( font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper',
					 color_codes=True, rc=None )

			# Set up the grid
			n_nodes, n_cols = len( impacted_nodes ), 3
			n_rows = int( np.ceil( n_nodes / n_cols ) )

			# Set up the figure and axis
			fig, axes = plt.subplots( n_rows, n_cols, figsize=figsize, sharey=True, sharex=True )  # type: ignore
			axes = axes.flatten()

			return fig, axes, n_rows, n_cols

		fig, axes, n_rows, n_cols = setup_plot()

		def plot_per_node(node, idx):

			row_idx = idx // n_cols
			col_idx = idx % n_cols
			ax = axes[idx]

			# Calculate the ROC curve and AUC
			def get_fpr_tpr_auc(pred_node, truth_node, technique: TechniqueNames, roc_auc):

				def get():
					nonlocal fpr, tpr
					# mask = ~truth_node.isnull()
					mask = ~np.isnan( truth_node )
					truth_notnull = truth_node[mask].to_numpy()

					if (len( truth_notnull ) > 0) and (np.unique( truth_notnull ).size == 2):
						fpr, tpr, _ = sklearn.metrics.roc_curve( truth_notnull, pred_node[mask] )
						return fpr, tpr
					return None, None

				fpr, tpr = get()
				return sns.lineplot( x=fpr, y=tpr, label=f'{technique} AUC = {roc_auc:.2f}', linewidth=2, ax=ax )

			# Plot the ROC curve
			lines, labels = [], []

			for technique_name in TechniqueNames.members():
				data = getattr( self, technique_name.lower() )
				technique = TechniqueNames[technique_name]

				line = get_fpr_tpr_auc( pred_node=data.pred[node], truth_node=truth[node], technique=technique,
										roc_auc=data.auc_acc_f1[node][EvaluationMetricNames.AUC.name] )
				lines.append( line.lines[-1] )
				labels.append( line.get_legend_handles_labels()[1][-1] )

			# Customize the plot
			ax.plot( [0, 1], [0, 1], linestyle='--', linewidth=2 )
			ax.set_xlabel( 'False Positive Rate', fontsize=fontsize,
						   labelpad=labelpad ) if row_idx == n_rows - 1 else ax.set_xticklabels( [] )
			ax.set_ylabel( 'True Positive Rate', fontsize=fontsize,
						   labelpad=labelpad ) if col_idx == 0 else ax.set_yticklabels( [] )
			ax.legend( loc='lower right', fontsize=12 )
			ax.set_xlim( [0.0, 1.0] )
			ax.set_ylim( [0.0, 1.05] )

			leg = ax.legend( lines, labels, loc='lower right', fontsize=fontsize, title=node )
			plt.setp( leg.get_title(), fontsize=fontsize )

			# Set the background color of the plot to gray if the node is a parent node
			if node in list_parent_nodes:
				ax.set_facecolor( 'xkcd:light grey' )

			fig.suptitle( 'ROC Curves', fontsize=int( 1.5 * fontsize ), fontweight='bold' )
			plt.tight_layout()

		def postprocess():
			# Remove any empty plots in the grid
			for empty_idx in range( idx + 1, n_rows * n_cols ):
				axes[empty_idx].axis( 'off' )

			plt.tight_layout()

		# Loop through each disease and plot the ROC curve
		for idx, node in enumerate( self.nodes.IMPACTED ):
			plot_per_node( node, idx )

		# Postprocess the plot
		postprocess()

		# Save the plot
		if save_figure:
			file_path = self.config.output.path / f'figures/roc_curve_all_datasets/{self.config.hyperparameter_tuning.threshold_technique}/roc_curve_all_datasets.png'
			LoadSaveFile( file_path ).save( data=fig )



'''
class TaxonomyXRV:

	def __init__(self, config: 'Settings', seed: int=10):

		self.hyperparameters = None
		self.config         : 'Settings'                  = config
		self.train          : Optional[Data]                      = None
		self.test           : Optional[Data]                      = None
		self.model          : Optional[torch.nn.Module]           = None
		self.dataset         : Optional[xrv.datasets.CheX_Dataset] = None

		technique_name = config.technique_name or EvaluationMetricNames.LOSS
		self.save_path : str = f'details/{config.DEFAULT_FINDING_FOLDER_NAME}/{technique_name}'

		# Setting the seed
		self.setting_random_seeds_for_pytorch(seed=seed)

	@staticmethod
	def measuring_bce_loss(p, y):
		return -( y * np.log(p) + (1 - y) * np.log(1 - p) )

	@staticmethod
	def equations_sigmoidprime(p):
		""" Refer to Eq. (10) in the paper draft """
		return p*(1-p)

	@staticmethod
	def setting_random_seeds_for_pytorch(seed=10):
		np.random.seed(seed)
		torch.manual_seed(seed)
		if USE_CUDA:
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark     = False

	def threshold(self, data_mode = DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		exp_stage_list   = [ExperimentStageNames.ORIGINAL.name       , ExperimentStageNames.NEW.name]
		thresh_tech_list = [ThreshTechList.PRECISION_RECALL.name, ThreshTechList.ROC.name]

		df = pd.DataFrame(  index   = data.ORIGINAL.threshold.index ,
							columns = pd.MultiIndex.from_product([thresh_tech_list, exp_stage_list]) )

		for th_tqn in [ThreshTechList.ROC.value , ThreshTechList.PRECISION_RECALL.value]:
			df[ (th_tqn, ExperimentStageNames.ORIGINAL.name)] = data.ORIGINAL.threshold[th_tqn]
			df[ (th_tqn, ExperimentStageNames.NEW.name)] = data.NEW     .threshold[th_tqn]

		return df.replace(np.nan, '')

	@staticmethod
	def accuracy_per_node(data, node, experimentStage, threshold_technique) -> float:

		findings = getattr(data,experimentStage)

		if node in data.labels.nodes.non_null:
			thresh = findings.threshold[threshold_technique][node]
			pred   = (findings.pred [node] >= thresh)
			truth  = (findings.truth[node] >= 0.5 )
			return (pred == truth).mean()

		return np.nan

	def accuracy(self, data_mode=DataModes.TRAIN) -> pd.DataFrame:

		data 	    = getattr(self, data_mode.value)
		pathologies = self.model.pathologies
		columns 	= pd.MultiIndex.from_product([ThreshTechList, data.list_findings_names])
		df 			= pd.DataFrame(index=pathologies , columns=columns)

		for node in pathologies:
			for xf in columns:
				df.loc[node, xf] = TaxonomyXRV.accuracy_per_node(data=data, node=node, threshold_technique=xf[0], experimentStage=xf[1])

		return df.replace(np.nan, '')

	def findings_per_node(self, node, data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		# Getting the hierarchy_penalty for node
		hierarchy_penalty = pd.DataFrame(columns=ThreshTechList.members())
		for x in ThreshTechList:
			hierarchy_penalty[x] = data.hierarchy_penalty[x,node]

		# Getting Metrics for node
		metrics = pd.DataFrame()
		for m in EvaluationMetricNames.members():
			metrics[m] = self.get_metric(metric=m, data_mode=data_mode).T[node]

		return Hierarchy.OUTPUT(hierarchy_penalty=hierarchy_penalty, metrics=metrics, data=data)

	def findings_per_node_iterator(self, data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		return iter( [ self.findings_per_node(node)  for node in data.Hierarchy_cls.parent_dict.keys() ] )

	def findings_per_node_with_respect_to_their_parent(self, node, thresh_technic: ThreshTechList = ThreshTechList.ROC, data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		N = data.Hierarchy_cls.graph.nodes
		parent_child = data.Hierarchy_cls.parent_dict[node] + [node]

		df = pd.DataFrame(index=N[node][ ExperimentStageNames.ORIGINAL]['data'].index, columns=pd.MultiIndex.from_product([parent_child, ['truth' , 'pred' , 'loss'], ExperimentStageNames.members()]))

		for n in parent_child:
			for dtype in ['truth' , 'pred' , 'loss']:
				df[ (n, dtype, ExperimentStageNames.ORIGINAL)] = N[n][ ExperimentStageNames.ORIGINAL]['data'][dtype].values
				df[ (n , dtype, ExperimentStageNames.NEW)]     = N[n][ ExperimentStageNames.NEW]['data'][thresh_technic][dtype].values

			df[(n, 'hierarchy_penalty', ExperimentStageNames.NEW)] = N[n]['hierarchy_penalty'][thresh_technic].values

		return df.round(decimals=3).replace(np.nan, '', regex=True)


	def save_metrics(self):

		for metric in EvaluationMetricNames.members() + ['Threshold']:

			# Saving the data
			path = self.config.PATH_LOCAL.joinpath( f'{self.save_path}/{metric}.xlsx' )

			# Create a new Excel writer
			with pd.ExcelWriter(path, engine='openpyxl') as writer:

				# Loop through the data modes
				for data_mode in DataModes:
					self.get_metric(metric=metric, data_mode=data_mode).to_excel(writer, sheet_name=data_mode.value)

			# Save the Excel file
			# writer.save()

	def get_metric(self, metric: EvaluationMetricNames=EvaluationMetricNames.AUC, data_mode: DataModes=DataModes.TRAIN) -> pd.DataFrame:

		data: 'Data' = self.train if data_mode == DataModes.TRAIN else self.test

		column_names = data.labels.nodes.impacted

		columns = pd.MultiIndex.from_product([ThreshTechList, ExperimentStageNames.members()], names=['threshold_technique', 'WR'])
		df = pd.DataFrame(index=data.ORIGINAL.pathologies, columns=columns)

		for x in ThreshTechList:
			if hasattr(data.ORIGINAL, 'metrics'):
				df[x, ExperimentStageNames.ORIGINAL] = data.ORIGINAL.metrics[x].T[metric.name]
			if hasattr(data.NEW, 'metrics'):
				df[x, ExperimentStageNames.NEW] = data.NEW     .metrics[x].T[metric.name]

		df = df.apply(pd.to_numeric, errors='ignore').round(3).replace(np.nan, '')

		return df.T[column_names].T

	@staticmethod
	def get_data_and_model(config):

		# Load the model
		model = LoadModelXRV(config).load().model

		# Load the data
		LD = LoadChestXrayDatasets.load( config=config )
		LD.load()

		return LD.train, LD.test, model, LD.dataset_full

	@classmethod
	def run_full_experiment(cls, technique_name=TechniqueNames.LOSS, seed=10, **kwargs):

		# Getting the user arguments
		config = get_settings( jupyter=True, **kwargs )

		# Initializing the class
		FE = cls(config=config, seed=seed)

		# Loading train/test data as well as the pre-trained model
		FE.train, FE.test, FE.model, FE.dataset = cls.get_data_and_model(FE.config)

		param_dict = {key: getattr(FE, key) for key in ['model', 'config']}

		# Measuring the ORIGINAL metrics (predictions and losses, thresholds, aucs, etc.)
		FE.train = CalculateOriginalFindings.get_updated_data(data=FE.train, **param_dict)
		FE.test  = CalculateOriginalFindings.get_updated_data(data=FE.test , **param_dict)

		# Calculating the hyperparameters
		FE.hyperparameters = HyperParameterTuning.get_updated_data(data=FE.train, **param_dict)

		# Adding the new findings to the graph nodes
		FE.train.Hierarchy_cls.update_graph(hyperparameters=FE.hyperparameters)
		FE.test. Hierarchy_cls.update_graph(hyperparameters=FE.hyperparameters)

		# Measuring the updated metrics (predictions and losses, thresholds, aucs, etc.)
		param_dict = {key: getattr(FE, key) for key in ['model', 'config', 'hyperparameters']}
		FE.train = CalculateNewFindings.get_updated_data(data=FE.train, technique_name=technique_name, **param_dict)
		FE.test  = CalculateNewFindings.get_updated_data(data=FE.test , technique_name=technique_name, **param_dict)

		# Saving the metrics: AUC, threshold, accuracy
		FE.save_metrics()

		return FE

	@staticmethod
	def loop_run_full_experiment():
		for datasetName in DatasetNames.members():
			for technique_name in TechniqueNames:
				TaxonomyXRV.run_full_experiment(technique_name=technique_name, datasetName=datasetName)

	@classmethod
	def get_merged_data(cls, data_mode=DataModes.TEST, technique_name='logit', threshold_technique='DEFAULT', datasets_list=None):  # type: (str, str, str, list) -> Tuple[DataMerged, DataMerged]

		if datasets_list is None:
			datasets_list = DatasetNames

		def get(method: ExperimentStageNames) -> DataMerged:
			data = defaultdict(list)
			for datasetName in datasets_list:
				a1 = cls.run_full_experiment(technique_name=technique_name, datasetName=datasetName)

				metric = getattr( getattr(a1,data_mode),method.value)
				data['pred'].append(metric.pred[threshold_technique] if method == ExperimentStageNames.NEW else metric.pred)
				data['truth'].append(metric.truth)
				data['yhat'].append(data['pred'][-1] >= metric.metrics[threshold_technique].T['Threshold'].T )
				data['list_nodes_impacted'].append(getattr(a1, data_mode).labels.nodes.impacted)

			return DataMerged(data)

		baseline = get(ExperimentStageNames.ORIGINAL)
		proposed = get(ExperimentStageNames.NEW)

		return baseline, proposed

	@classmethod
	def get_all_metrics(cls, datasets_list=DatasetNames.members(), data_mode=DataModes.TEST, threshold_technique=ThreshTechList.DEFAULT, jupyter=True, **kwargs):  # type: (List[DatasetNames], DataModes, ThreshTechList, bool, dict) -> MetricsAllTechniques

		config = get_settings(jupyter=jupyter, **kwargs)
		save_path = pathlib.Path(f'tables/metrics_all_datasets/{threshold_technique}')

		def apply_to_approach(technique_name: TechniqueNames) -> Metrics:

			baseline, proposed = cls.get_merged_data(data_mode=data_mode, technique_name=technique_name, threshold_technique=threshold_technique, datasets_list=datasets_list)

			def get_auc_acc_f1(Union[Node, str], data: DataMerged):

				# Finding the indices where the truth is not nan
				non_null = ~np.isnan( data.truth[node] )
				truth_notnull = data.truth[node][non_null].to_numpy()

				if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
					data.auc_acc_f1[node][EvaluationMetricNames.AUC.name] = sklearn.metrics.roc_auc_score(data.truth[node][non_null], data.yhat[node][non_null])
					data.auc_acc_f1[node][EvaluationMetricNames.ACC.name] = sklearn.metrics.accuracy_score(data.truth[node][non_null], data.yhat[node][non_null])
					data.auc_acc_f1[node][EvaluationMetricNames.F1.name]  = sklearn.metrics.f1_score(data.truth[node][non_null], data.yhat[node][non_null])

			def get_p_value_kappa_cohen_d_bf10(df, node):  # type: (pd.DataFrame, str) -> None

				# Perform the independent samples t-test
				df.loc['t_stat',node], df.loc['p_value',node] = stats.ttest_ind( baseline.yhat[node], proposed.yhat[node])

				# kappa inter rater metric
				df.loc['kappa',node] = sklearn.metrics.cohen_kappa_score(baseline.yhat[node], proposed.yhat[node])

				df_ttest = pg.ttest(baseline.yhat[node], proposed.yhat[node])
				df.loc['power',node]   = df_ttest['power'].values[0]
				df.loc['cohen-d',node] = df_ttest['cohen-d'].values[0]
				df.loc['BF10',node]    = df_ttest['BF10'].values[0]

			metrics_comparison = pd.DataFrame(columns=baseline.pred.columns, index=['kappa', 'p_value', 't_stat', 'power', 'cohen-d','BF10'])
			# auc_acc_f1_baseline = pd.DataFrame (columns=baseline.pred.columns, index=EvaluationMetricNames.members())
			# auc_acc_f1_proposed = pd.DataFrame (columns=baseline.pred.columns, index=EvaluationMetricNames.members())

			for node in baseline.pred.columns:
				get_auc_acc_f1(node, baseline)
				get_auc_acc_f1(node, proposed)
				get_p_value_kappa_cohen_d_bf10(metrics_comparison, node)

			return Metrics(metrics_comparison=metrics_comparison, baseline=baseline, proposed=proposed, config=config, technique_name=technique_name)

		def get_auc_acc_f1_merged(logit, loss):  # type: (Metrics, Metrics) -> pd.DataFrame
			columns = pd.MultiIndex.from_product([EvaluationMetricNames.members(), TechniqueNames.members()])
			auc_acc_f1 = pd.DataFrame(columns=columns)

			for metric in EvaluationMetricNames.members():
				auc_acc_f1[metric] = pd.DataFrame( dict(baseline=loss.baseline.auc_acc_f1.T[metric],
														loss=loss.proposed.auc_acc_f1.T[metric],
														logit=logit.proposed.auc_acc_f1.T[metric] ))

			return auc_acc_f1

		if config.do_metrics == 'calculate':
			logit 	   = apply_to_approach(TechniqueNames.LOGIT)
			loss  	   = apply_to_approach(TechniqueNames.LOSS)
			auc_acc_f1 = get_auc_acc_f1_merged(logit, loss)

			# Saving the metrics locally
			LoadSaveFindings(config, save_path / 'logit_metrics.csv').save(logit.metrics_comparison[logit.baseline.list_nodes_impacted].T)
			LoadSaveFindings(config, save_path / 'logit.pkl').save(logit)

			LoadSaveFindings(config, save_path / 'loss_metrics.csv').save(loss.metrics_comparison[loss.baseline.list_nodes_impacted].T)
			LoadSaveFindings(config, save_path / 'loss.pkl').save(loss)

			LoadSaveFindings(config, save_path / 'auc_acc_f1.xlsx').save(auc_acc_f1, index=True)

		else:
			load_lambda = lambda x, **kwargs: LoadSaveFindings(config, save_path.joinpath(x)).load(
					**kwargs)
			logit 	   = load_lambda('logit.pkl')
			loss 	   = load_lambda('loss.pkl')
			auc_acc_f1 = load_lambda('auc_acc_f1.xlsx', index_col=0, header=[0, 1])

		return MetricsAllTechniques(loss=loss, logit=logit, auc_acc_f1=auc_acc_f1, threshold_technique=threshold_technique, datasets_list=datasets_list, data_mode=data_mode)

	@classmethod
	def get_all_metrics_all_thresh_techniques(cls, datasets_list: list[str]=['CheX', 'NIH', 'PC'], data_mode: str=DataModes.TEST) -> MetricsAllTechniqueThresholds:

		output = {}
		for x in tqdm(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
			output[x] = TaxonomyXRV.get_all_metrics(datasets_list=datasets_list, data_mode=data_mode, threshold_technique=x)

		return MetricsAllTechniqueThresholds(**output)

'''

'''
class Tables:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	def get_metrics_per_thresh_techniques(self, save_table=True, data_mode=DataModes.TEST.value, threshold_technique='DEFAULT'):

		from taxonomy.utilities.utils import TaxonomyXRV

		save_path = self.config.PATH_LOCAL.joinpath(
			f'tables/metrics_per_dataset/{threshold_technique}/metrics_{data_mode}.xlsx')

		def save(metricsa):

			save_path.parent.mkdir(parents=True, exist_ok=True)

			# Create a new Excel writer
			with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
				# Write each metric to a different worksheet
				for m in EvaluationMetricNames:
					getattr(metricsa, m.value).to_excel(writer, sheet_name=m.name)

		def get():
			columns = pd.MultiIndex.from_product([DatasetNames.members(), TechniqueNames.members()],
			                                     names=['dataset_full', 'technique_name'])
			AUC = pd.DataFrame(columns = columns)
			F1  = pd.DataFrame(columns = columns)
			ACC = pd.DataFrame(columns = columns)

			LOGIT    = TechniqueNames.LOGIT.modelName
			LOSS     = TechniqueNames.LOSS.modelName
			BASELINE = TechniqueNames.BASELINE.name

			for dt in DatasetNames.members():
				df = TaxonomyXRV.run_full_experiment(technique_name=TechniqueNames.LOGIT, datasetName=dt)
				AUC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[ EvaluationMetricNames.ACC.name]

				AUC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[threshold_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[threshold_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[threshold_technique].loc[ EvaluationMetricNames.ACC.name]

				df = TaxonomyXRV.run_full_experiment(technique_name=TechniqueNames.LOSS, datasetName=dt)
				AUC[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOSS)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[EvaluationMetricNames.F1.name]
				ACC[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[threshold_technique].loc[ EvaluationMetricNames.ACC.name]

			AUC = AUC.apply(pd.to_numeric).round(3).replace(np.nan, '')
			F1  = F1.apply( pd.to_numeric).round(3).replace(np.nan, '')
			ACC = ACC.apply(pd.to_numeric).round(3).replace(np.nan, '')

			# region load Data & Model
			@dataclass
			class Metrics:
				AUC: pd.DataFrame
				ACC: pd.DataFrame
				F1: pd.DataFrame

			return Metrics(AUC=AUC, F1=F1, ACC=ACC)

		metrics = get()

		if save_table:
			save(metrics)

		return metrics

	@staticmethod
	def get_dataset_unfiltered(**kwargs):
		config = get_settings(**kwargs)
		return LoadChestXrayDatasets( config=config, datasetInfo=config.dataset.datasetInfoList[0]).get_dataset_unfiltered()

	def get_table_datasets_samples(self, save_table=True):

		save_path = pathlib.Path('tables/metrics_all_datasets/table_datasets_samples.csv')

		def get_PA_AP(mode: str, dname: str) -> pd.Series:
			# nonlocal mode, dname

			def combine_PA_AP(row):
				return '' if (not row.PA) and (not row.AP) else f"{row.PA}/{row.AP}"

			df2 = pd.DataFrame(columns=['PA', 'AP'])
			for views in ['PA', 'AP']:

				# Getting the dataset_full for a specific view
				LD = Tables.get_dataset_unfiltered( datasetName=dname, views=views )
				df2[views] = LD.dataset_full.labels.sum( axis=0 ).astype( int ).replace( 0, '' )

				# Adding the Total row
				df2.loc['Total', views] = LD.dataset_full.labels.shape[0]

			return df2.apply(combine_PA_AP, axis=1)

		cln_list = [ExperimentStageNames.members(), DatasetNames.members()]
		columns = pd.MultiIndex.from_product(cln_list, names=['mode', 'dataset_full'])
		df = pd.DataFrame(columns=columns)

		for mode, dname in itertools.product(*cln_list):
			df[(mode, dname)] = get_PA_AP(mode=mode, dname=dname)

		if save_table:
			LoadSaveFile(file_path).save(data=df)

		return df


class Visualize:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	@staticmethod
	def plot_class_relationships(config: 'Settings', method: str = 'TSNE',
	                             data_mode: DataModes = DataModes.TEST, feature_maps: Optional[np.ndarray] = None,
	                             labels: pd.DataFrame = None) -> None:

		path_main = config.PATH_LOCAL.joinpath(f'{config.DEFAULT_FINDING_FOLDER_NAME}/class_relationship')

		def get_reduced_features() -> np.ndarray:
			if method.upper() == 'UMAP':
				reducer = umap.UMAP()
				return reducer.fit_transform(feature_maps)

			elif method.upper() == 'TSNE':
				from sklearn.manifold import TSNE
				return TSNE(n_components=2, random_state=42).fit_transform(feature_maps)

		def do_plot(df_truth: pd.DataFrame) -> None:

			def save_plot():

				# output path
				filename = f'class_relationship_{method}'
				path = path_main.joinpath(method)
				path.mkdir(parents=True, exist_ok=True)

				# Save the plot
				for ft in ['png', 'eps', 'svg', 'pdf']:
					plt.savefig(path.joinpath(f'{filename}.{ft}'), format=ft, dpi=300)

			colors = plt.cm.get_cmap('tab20', max(18, len(df_truth.columns)))
			_, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
			axes = axes.flatten()

			for i, node in enumerate(df_truth.columns):
				class_indices = df_truth[df_truth[node].eq(1)].index.to_numpy()
				axes[i].scatter(X_embedded[:, 0], X_embedded[:, 1], color='lightgray', alpha=0.2)
				axes[i].scatter(X_embedded[class_indices, 0], X_embedded[class_indices, 1], c=[colors(i)], alpha=0.5,
				                s=20)
				axes[i].set_title(node)

			plt.suptitle(f"{method} Visualization for {config.datasetName} dataset_full")

			# Save the plot
			save_plot()

			plt.show()

		# Get feature maps
		if feature_maps is None:
			feature_maps, labels, list_non_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)
			labels = labels[list_non_null_nodes]

		# Get Reduced features
		X_embedded = get_reduced_features()

		# Plot
		do_plot(df_truth=labels)

	@staticmethod
	def plot_class_relationships_objective_function(data_mode, datasetName):

		config = get_settings(datasetName=datasetName)

		feature_maps, labels, list_non_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)

		for method in ['UMAP', 'TSNE']:
			Visualize.plot_class_relationships(config=config, method=method, data_mode=data_mode,
			                                   feature_maps=feature_maps, labels=labels[list_non_null_nodes])

	def plot_metrics_all_thresh_techniques(self, save_figure=False):

		def get_metrics():
			from taxonomy.utilities.utils import TaxonomyXRV

			columns = pd.MultiIndex.from_product([EvaluationMetricNames.members(), TechniqueNames.members()])
			metric_df = {}
			for threshold_technique in ThreshTechList:
				output = TaxonomyXRV.get_all_metrics(datasets_list=DatasetNames.members(),
				                                     data_mode=DataModes.TEST,
				                                     threshold_technique=threshold_technique)
				metric_df[threshold_technique] = pd.DataFrame(columns=columns)
				for metric in EvaluationMetricNames.members():
					df = pd.DataFrame(dict(baseline=output.loss.baseline.auc_acc_f1.T[metric],
					                       loss=output.loss.proposed.auc_acc_f1.T[metric],
					                       logit=output.logit.proposed.auc_acc_f1.T[metric]))
					metric_df[threshold_technique][metric] = df.T[output.list_nodes_impacted].T

			return metric_df

		metric_df = get_metrics()
		fig, axes = plt.subplots(3, 3, figsize=(21, 21), sharey=True, sharex=True)  # type: ignore
		sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True,
					  rc=None)

		params = dict(legend=False, fontsize=16, kind='barh')
		for i, threshold_technique in enumerate(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
			metric_df[threshold_technique][EvaluationMetricNames.ACC.name].plot(ax=axes[i, 0],
																			 xlabel=EvaluationMetricNames.ACC.name,
																			 ylabel=threshold_technique, **params)
			metric_df[threshold_technique][EvaluationMetricNames.AUC.name].plot(ax=axes[i, 1],
																			 xlabel=EvaluationMetricNames.AUC.name,
																			 ylabel=threshold_technique, **params)
			metric_df[threshold_technique][EvaluationMetricNames.F1.name].plot(ax=axes[i, 2],
																			xlabel=EvaluationMetricNames.F1.name,
																			ylabel=threshold_technique, **params)

		plt.legend(loc='lower right', fontsize=16)
		plt.tight_layout()

		if save_figure:
			save_path = self.config.output.path / 'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/metrics_AUC_ACC_F1.png'
			LoadSaveFile(file_path).save(data=fig, file_format=['png', 'eps', 'svg', 'pdf'])


	@staticmethod
	def plot_metrics(config: 'Settings', metrics: pd.DataFrame, threshold_technique: ThreshTechList , save_figure=True, figsize=(21, 7), font_scale=1.8, fontsize=20):


		def save_plot():
			save_path = config.PATH_LOCAL.joinpath(f'figures/auc_acc_f1_all_datasets/{threshold_technique}/')
			save_path.mkdir(parents=True, exist_ok=True)
			for ft in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig( save_path.joinpath( f'metrics_AUC_ACC_F1.{ft}' ), format=ft, dpi=300 )

		def barplot():
			fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)  # type: ignore
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True, rc=None)

			params = dict(legend=False, fontsize=16, kind='barh')

			metrics[EvaluationMetricNames.ACC.name].plot(ax = axes[0], title = EvaluationMetricNames.ACC.name, **params)
			metrics[EvaluationMetricNames.AUC.name].plot(ax = axes[1], title = EvaluationMetricNames.AUC.name, **params)
			metrics[EvaluationMetricNames.F1.name].plot(ax  = axes[2], title = EvaluationMetricNames.F1.name , **params)
			plt.legend(loc='upper right', fontsize=16)
			plt.tight_layout()


		def heatmap():
			sns.set(font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper', color_codes=True, rc=None)

			fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)  # type: ignore
			params = dict(annot=True, fmt=".3f", linewidths=.5, cmap='YlGnBu', cbar=False, annot_kws={"size": fontsize})

			for i, m in enumerate(EvaluationMetricNames.members()):
				sns.heatmap(data=metrics[m],ax=axes[i], **params)
				axes[i].set_title(m, fontsize=int(1.5*fontsize), fontweight='bold')
				axes[i].tick_params(axis='both', which='major', labelsize=fontsize)

			plt.tight_layout()

		heatmap()

		if save_figure:
			save_plot()

'''
