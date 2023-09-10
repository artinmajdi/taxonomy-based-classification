import concurrent.futures
import concurrent.futures
import contextlib
import multiprocessing
import pathlib
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pingouin as pg
import sklearn
import torch
import torchxrayvision as xrv
from hyperopt import fmin, hp, tpe
from scipy import stats
from tqdm import tqdm

from taxonomy.utilities.data import Data, LoadChestXrayDatasets, Nodes
from taxonomy.utilities.findings import Findings, HyperParameters, HyperPrametersNode, Metrics, ModelOutputs, \
	ModelOutputsNode
from taxonomy.utilities.model import LoadModelXRV, ModelType
from taxonomy.utilities.params import DataModes, DatasetNames, EvaluationMetricNames, ExperimentStageNames, \
	ParentMetricToUseNames, TechniqueNames, ThreshTechList
from taxonomy.utilities.settings import get_settings, Settings

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

@dataclass
class CalculateOriginalFindings:
	config: Settings
	data  : Data
	model : ModelType

	def __post_init__(self):
		self.findings = Findings( config = self.config,
								  data   = self.data,
								  model  = self.model,
								  experiment_stage = ExperimentStageNames.ORIGINAL )

	def calculate(self) -> 'CalculateOriginalFindings':

		pathologies        = self.model.pathologies
		loss_function      = self.config.training.criterion.function
		batches_to_process = self.config.training.batches_to_process
		model_outputs      = ModelOutputs()
		self.model.eval()

		def process_one_batch(batch_data_in) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

			def get_truth_and_predictions() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

				# Sample data and its corresponding labels
				images = batch_data_in["img"].to(device)
				truth  = batch_data_in["lab"].to(device)

				# Feeding the samples into the model
				logit = self.findings.model( images )
				pred  = torch.sigmoid(logit) # .detach().numpy()

				return pred , logit , truth

			# Getting the true and predicted labels
			pred_batch, logit_batch, truth_batch = get_truth_and_predictions()

			# Getting the index of the samples for this batch
			index = batch_data_in['idx'].detach().cpu().numpy()

			def get_loss_per_sample_batch() -> pd.DataFrame:
				nonlocal pred_batch, truth_batch, index

				l_batch = pd.DataFrame(columns=pathologies, index=index)

				for ix, lbl_name in enumerate(pathologies):

					# This skips the empty labels (i.e., the labels that are not in both the dataset & model)
					if len(lbl_name) == 0: continue

					task_truth, task_pred = truth_batch[:, ix].double() , pred_batch[:, ix].double()

					# Calculating the loss per sample
					l_batch[lbl_name] = loss_function(task_pred, task_truth).detach().cpu().numpy()

				return l_batch

			loss_per_sample_batch = get_loss_per_sample_batch()

			# Converting the outputs and targets to dataframes
			def to_df(data_tensor: torch.Tensor):
				return pd.DataFrame(data_tensor.detach().cpu(), columns=self.model.pathologies, index=index)

			return to_df(pred_batch), to_df(logit_batch), to_df(truth_batch), loss_per_sample_batch

		def looping_over_all_batches() -> None:

			for batch_idx, batch_data in enumerate(self.data.data_loader):

				if batch_idx >= batches_to_process: break

				pred, logit, truth, loss = process_one_batch(batch_data)

				# Appending the results for this batch to the results
				model_outputs.pred  = pd.concat([model_outputs.pred , pred ])
				model_outputs.logit = pd.concat([model_outputs.logit, logit])
				model_outputs.truth = pd.concat( [model_outputs.truth, truth] )
				model_outputs.loss  = pd.concat( [model_outputs.loss , loss] )

		with torch.no_grad():

			# Looping over all batches
			looping_over_all_batches()

			# Adding model outputs to the findings.
			self.findings.model_outputs = model_outputs

			self.findings.metrics = Metrics.calculate(config=self.config, REMOVE_NULL=True, model_outputs=self.findings.model_outputs)

		return self

	def save(self) -> 'CalculateOriginalFindings':

		# Saving model_outputs to disc
		self.findings.model_outputs.save(config=self.config, experiment_stage=ExperimentStageNames.ORIGINAL)

		# Calculating the metrics & saving them to disc
		self.findings.metrics.save(config=self.config, experiment_stage=ExperimentStageNames.ORIGINAL)

		return self

	def load(self) -> 'CalculateOriginalFindings':

		# Loading model_outputs from disc
		self.findings.model_outputs = ModelOutputs().load(config=self.config, experiment_stage=ExperimentStageNames.ORIGINAL)

		# Loading the metrics from disc
		self.findings.metrics = Metrics().load(config=self.config, experiment_stage=ExperimentStageNames.ORIGINAL)

		return self


@dataclass
class CalculateNewFindings:
	findings_original: Findings
	hyper_parameters : HyperParameters

	def __post_init__(self):
		self.config = self.findings_original.config

		self.findings = Findings( config = self.findings_original.config,
								  data   = self.findings_original.data,
								  model  = self.findings_original.model,
								  experiment_stage = ExperimentStageNames.NEW)


	@staticmethod
	def calculate_for_node(findings_original: Findings, hyper_parameters_node: HyperPrametersNode, node: str) -> ModelOutputsNode:

		config: Settings = findings_original.config
		parent_node: str = findings_original.data.nodes.get_parent_of(child=node)

		if parent_node is None:
			return findings_original.model_outputs[node]

		def _calculate_penalty_for_node() -> pd.Series:
			""" Method to calculate hierarchical penalty for a node. """

			# The hyper_parameters for the current node
			MULTIPLIER: float = hyper_parameters_node.MULTIPLIER
			ADDITIVE  : float = hyper_parameters_node.ADDITIVE

			# Getting the parent node's model_outputs data
			pdata: ModelOutputsNode = findings_original.model_outputs[parent_node]
			pthreshold : float = findings_original.metrics.THRESHOLD[parent_node]

			def parent_exist():
				""" Method to check if the parent class exist. """
				parent_metric_to_use: ParentMetricToUseNames = config.technique.parent_metric_to_use

				if parent_metric_to_use == ParentMetricToUseNames.PRED:
					return pdata.pred >= pthreshold

				elif parent_metric_to_use == ParentMetricToUseNames.TRUTH:
					return pdata.truth >= 0.5

				elif parent_metric_to_use == ParentMetricToUseNames.NONE:
					return pd.Series(True, pdata.truth.index)

			# Calculating the initial hierarchy_penalty based on "a", "b" and "technique_name"
			if config.technique_name == TechniqueNames.LOGIT:
				hierarchy_penalty = MULTIPLIER * pdata.logit

			elif config.technique_name == TechniqueNames.LOSS:
				hierarchy_penalty = MULTIPLIER * pdata.loss + ADDITIVE
			else:
				raise NotImplementedError(f"Technique {config.technique_name} not implemented")

			# Setting the hierarchy_penalty to one for samples where the parent class exist, because we can not infer any information from those samples.
			hierarchy_penalty[parent_exist()] = 1.0

			# Setting the hierarchy_penalty to 1.0 for samples where we don't have the truth label for parent class.
			hierarchy_penalty[pdata.truth.isnull()] = 1.0

			return hierarchy_penalty

		data: ModelOutputsNode = findings_original.model_outputs[node]
		penalty: pd.Series     = _calculate_penalty_for_node()

		def do_loss():
			nonlocal data

			# Measuring the new loss values
			data.loss = penalty * data.loss

			# Calculating the loss gradient to find the direction of changes
			loss_gradient = -data.truth / (data.pred + 1e-7) + (1 - data.truth) / (1 - data.pred + 1e-7)

			# Calculating the new predicted probability
			data.pred = np.exp( -data.loss )

			condition = loss_gradient >= 0
			data.pred[condition] = 1 - data.pred[condition]

		def do_logit():
			nonlocal data

			# Measuring the new loss values
			data.logit = penalty + data.logit

			# Calculating the new predicted probability
			data.pred = 1 / (1 + np.exp( -data.logit ))

		if  config.technique_name == TechniqueNames.LOGIT:
			do_logit()

		elif config.technique_name == TechniqueNames.LOSS:
			do_loss()

		return data


	@classmethod
	def calculate(cls, findings_original: Findings, hyper_parameters: HyperParameters) -> 'CalculateNewFindings':

		self = cls(findings_original=findings_original, hyper_parameters=hyper_parameters)

		# Initializing the model_outputs with empty dataframes
		truth = self.findings_original.model_outputs.truth
		self.findings.model_outputs = ModelOutputs.initialize(columns=truth.columns, index=truth.index)

		for node in self.findings_original.data.nodes.classes:
			self.findings.model_outputs[node] = self.calculate_for_node(node=node, findings_original=findings_original, hyper_parameters_node=hyper_parameters[node])


		# Calculating the metrics
		self.findings.metrics = Metrics.calculate(config=self.config, REMOVE_NULL=True, model_outputs=self.findings.model_outputs)

		return self

	def save(self) -> 'CalculateNewFindings':

		# Saving model_outputs to disc
		self.findings.model_outputs.save(config=self.config, experiment_stage=ExperimentStageNames.NEW)

		# Calculating the metrics & saving them to disc
		self.findings.metrics.save(config=self.config, experiment_stage=ExperimentStageNames.NEW)

		return self

	def load(self) -> 'CalculateNewFindings':

		# Loading model_outputs from disc
		self.findings.model_outputs = ModelOutputs().load(config=self.config, experiment_stage=ExperimentStageNames.NEW)

		# Loading the metrics from disc
		self.findings.metrics = Metrics().load(config=self.config, experiment_stage=ExperimentStageNames.NEW)

		return self


@dataclass
class HyperParameterTuning:
	findings_original: Findings

	@staticmethod
	def calculate_per_node(data: Data, config: Settings, hyper_parameters: Dict[str, pd.DataFrame], node: str, thresh_technique: ThreshTechList=ThreshTechList.DEFAULT) -> List[float]:

		def objective_function(args: Dict[str, float], hp_in) -> float:

			# Updating the hyper_parameters for the current node and thresholding technique
			hp_in[thresh_technique][node] = [args['a'], args['b']]

			data2: Data = CalculateNewFindings.calculate_per_node( node=node, data=data, config=config, hyper_parameters=hp_in, thresh_technique=thresh_technique)

			# Returning the error
			return 1 - data2.NEW.metrics[thresh_technique, node][config.optimization_metric.upper()]

		# Run the optimization
		best = fmin(
					fn        = lambda args : objective_function( args=args, hp_in=hyper_parameters),
					space     = dict(a=hp.uniform('a', -1, 1), b=hp.uniform('b', -4, 4)) ,	# Search space for the variables
					algo      = tpe.suggest ,     # Optimization algorithm (Tree-structured Parzen Estimator)
					max_evals = config.max_evals , # Maximum number of evaluations
					verbose   = True ,	          # Verbosity level (2 for detailed information)
					)

		return [ best['a'] , best['b'] ]


	@staticmethod
	def calculate(initial_hp, config, data):  # type: (Dict[str, pd.DataFrame], Settings, Data) -> Dict[str, pd.DataFrame]

		def extended_calculate_per_node(nt: List[str]) -> List[Union[str, float]]:
			hyperparameters_updated = HyperParameterTuning.calculate_per_node( node=nt[0], thresh_technique=nt[1],data=data, config=config, hyper_parameters=deepcopy(initial_hp))
			return [nt[0], nt[1]] + hyperparameters_updated # type: ignore

		def update_hyperparameters(results_in):  # type: ( List[Tuple[str, str, List[float]]] ) -> Dict[str, pd.DataFrame]
			# Creating a copy of initial hyper_parameters
			hp_in = initial_hp.copy()

			# Updating hyper_parameters with new findings
			for r_in in results_in:
				hp_in[r_in[1]][r_in[0]] = r_in[2]

			return hp_in

		stop_requested = False

		# Set the maximum number of open files
		# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))

		PARALLELIZE = config.parallelization_technique

		with contextlib.suppress(KeyboardInterrupt):

			if PARALLELIZE == 0:
				results = [extended_calculate_per_node(node_thresh) for node_thresh in data.labels.nodes.node_thresh_tuple]

			elif PARALLELIZE == 1:
				multiprocessing.set_start_method('spawn', force=True)
				with multiprocessing.Pool(processes=4) as pool:  # Set the number of worker processes here
					results = pool.map(extended_calculate_per_node, data.labels.nodes.node_thresh_tuple)

			elif PARALLELIZE == 2:
				with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

					# Submit each item in the input list to the executor
					futures = [executor.submit(extended_calculate_per_node, nt) for nt in data.labels.nodes.node_thresh_tuple]

					# Wait for all jobs to complete
					done, _ = concurrent.futures.wait(futures)

					# Fetch the results from the completed futures
					# results = [f.result() for f in done]

					results = []
					for f in concurrent.futures.as_completed(futures):
						if stop_requested: break
						r = f.result()
						results.append(r)

					# Manually close the futures
					for f in futures:
						f.cancel()

		return update_hyperparameters(results_in=results)


class TaxonomyXRV:

	def __init__(self, config: Settings, seed: int=10):

		self.hyper_parameters = None
		self.config         : Settings                  = config
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
	def accuracy_per_node(data, node, experimentStage, thresh_technique) -> float:

		findings = getattr(data,experimentStage)

		if node in data.labels.nodes.non_null:
			thresh = findings.threshold[thresh_technique][node]
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
				df.loc[node, xf] = TaxonomyXRV.accuracy_per_node(data=data, node=node, thresh_technique=xf[0], experimentStage=xf[1])

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

		data: Data = self.train if data_mode == DataModes.TRAIN else self.test

		column_names = data.labels.nodes.impacted

		columns = pd.MultiIndex.from_product([ThreshTechList, ExperimentStageNames.members()], names=['thresh_technique', 'WR'])
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

		# Calculating the hyper_parameters
		FE.hyper_parameters = HyperParameterTuning.get_updated_data(data=FE.train, **param_dict)

		# Adding the new findings to the graph nodes
		FE.train.Hierarchy_cls.update_graph(hyper_parameters=FE.hyper_parameters)
		FE.test. Hierarchy_cls.update_graph(hyper_parameters=FE.hyper_parameters)

		# Measuring the updated metrics (predictions and losses, thresholds, aucs, etc.)
		param_dict = {key: getattr(FE, key) for key in ['model', 'config', 'hyper_parameters']}
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
	def get_merged_data(cls, data_mode=DataModes.TEST, technique_name='logit', thresh_technique='DEFAULT', datasets_list=None):  # type: (str, str, str, list) -> Tuple[DataMerged, DataMerged]

		if datasets_list is None:
			datasets_list = DatasetNames

		def get(method: ExperimentStageNames) -> DataMerged:
			data = defaultdict(list)
			for datasetName in datasets_list:
				a1 = cls.run_full_experiment(technique_name=technique_name, datasetName=datasetName)

				metric = getattr( getattr(a1,data_mode),method.value)
				data['pred'].append(metric.pred[thresh_technique] if method == ExperimentStageNames.NEW else metric.pred)
				data['truth'].append(metric.truth)
				data['yhat'].append(data['pred'][-1] >= metric.metrics[thresh_technique].T['Threshold'].T )
				data['list_nodes_impacted'].append(getattr(a1, data_mode).labels.nodes.impacted)

			return DataMerged(data)

		baseline = get(ExperimentStageNames.ORIGINAL)
		proposed = get(ExperimentStageNames.NEW)

		return baseline, proposed

	@classmethod
	def get_all_metrics(cls, datasets_list=DatasetNames.members(), data_mode=DataModes.TEST, thresh_technique=ThreshTechList.DEFAULT, jupyter=True, **kwargs):  # type: (List[DatasetNames], DataModes, ThreshTechList, bool, dict) -> MetricsAllTechniques

		config = get_settings(jupyter=jupyter, **kwargs)
		save_path = pathlib.Path(f'tables/metrics_all_datasets/{thresh_technique}')

		def apply_to_approach(technique_name: TechniqueNames) -> Metrics:

			baseline, proposed = cls.get_merged_data(data_mode=data_mode, technique_name=technique_name, thresh_technique=thresh_technique, datasets_list=datasets_list)

			def get_auc_acc_f1(node: str, data: DataMerged):

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

		return MetricsAllTechniques(loss=loss, logit=logit, auc_acc_f1=auc_acc_f1, thresh_technique=thresh_technique, datasets_list=datasets_list, data_mode=data_mode)

	@classmethod
	def get_all_metrics_all_thresh_techniques(cls, datasets_list: list[str]=['CheX', 'NIH', 'PC'], data_mode: str=DataModes.TEST) -> MetricsAllTechniqueThresholds:

		output = {}
		for x in tqdm(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
			output[x] = TaxonomyXRV.get_all_metrics(datasets_list=datasets_list, data_mode=data_mode, thresh_technique=x)

		return MetricsAllTechniqueThresholds(**output)

