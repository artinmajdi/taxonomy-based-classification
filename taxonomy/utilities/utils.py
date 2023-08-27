import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing
import pathlib
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatch, singledispatchmethod, wraps
from typing import Any, Dict, List, Literal, Optional, overload, Tuple, Union

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import sklearn
import torch
from hyperopt import fmin, hp, tpe
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import stats
from tqdm import tqdm

import torchxrayvision as xrv
from taxonomy.utilities.data import Data, Findings, LoadChestXrayDatasets, Metrics, NodeData
from taxonomy.utilities.model import LoadModelXRV
from taxonomy.utilities.params import DataModes, DatasetNames, EvaluationMetricNames, ExperimentStageNames, \
	TechniqueNames, ThreshTechList
from taxonomy.utilities.settings import get_settings, Settings

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'




class CalculateOriginalFindings(LoadSaveFile):

	def __init__(self, data: Data, model: torch.nn.Module, config: Settings):
		self.findings_original = Findings(data=data, model=model, config=config)
		self.save_path_full    = config.output.path / 'details' / 'baseline' / 'findings_original.pkl'

	def calculate(self, data: Data,) -> Data:

		data               = self.findings_original.data
		model              = self.findings_original.model
		pathologies        = self.findings_original.model.pathologies
		loss_function      = self.findings_original.config.training.criterion.function
		batches_to_process = self.findings_original.config.training.batches_to_process
		modelFindings      = self.findings_original.modelFindings

		model.eval()

		def process_one_batch(batch_data_in):

			def get_truth_and_predictions():

				# Sample data and its corresponding labels
				images = batch_data_in["img"].to(device)
				truth  = batch_data_in["lab"].to(device)

				# Feeding the samples into the model
				logit = self.findings_original.model(images)
				pred  = torch.sigmoid(logit) # .detach().numpy()

				return pred , logit , truth

			# Getting the true and predicted labels
			pred_batch, logit_batch, truth_batch = get_truth_and_predictions()

			# Getting the index of the samples for this batch
			index = batch_data_in['idx'].detach().cpu().numpy()

			def get_loss_per_sample_batch():
				nonlocal pred_batch, truth_batch, index

				l_batch = pd.DataFrame(columns=pathologies, index=index)

				for ix, lbl_name in enumerate(pathologies):

					# This skips the empty labels (i.e. the labels that are not in both the dataset & model)
					if len(lbl_name) == 0: continue

					task_truth, task_pred = truth_batch[:, ix].double() , pred_batch[:, ix].double()

					# Calculating the loss per sample
					l_batch[lbl_name] = loss_function(task_pred, task_truth).detach().cpu().numpy()

				return l_batch

			loss_per_sample_batch = get_loss_per_sample_batch()

			# Converting the outputs and targets to dataframes
			# to_df = lambda data: pd.DataFrame(data.detach().cpu().numpy(), columns=model.pathologies, index=index)
			to_df = lambda data_tensor: pd.DataFrame(data_tensor.detach().cpu(), columns=model.pathologies, index=index)

			return to_df(pred_batch), to_df(logit_batch), to_df(truth_batch), loss_per_sample_batch

		def looping_over_all_batches() -> None:

			for batch_idx, batch_data in enumerate(data.data_loader):

				# This ensures that we only evaluate the data for a few batches. End the loop after n_batches_to_process
				if batch_idx >= batches_to_process: break

				pred, logit, truth, loss = process_one_batch(batch_data)

				# Appending the results for this batch to the results
				modelFindings.pred_values  = pd.concat([modelFindings.pred_values , pred ])
				modelFindings.logit_values = pd.concat([modelFindings.logit_values, logit])
				modelFindings.truth_values = pd.concat([modelFindings.truth_values, truth])
				modelFindings.loss_values  = pd.concat([modelFindings.loss_values , loss ])

		with torch.no_grad():

			# Looping over all batches
			looping_over_all_batches()

			# Measuring AUCs & Thresholds
			data.ORIGINAL = TaxonomyXRV.calculating_threshold_and_metrics(data.ORIGINAL)

		data.ORIGINAL.results = {key: getattr( data.ORIGINAL, key ) for key in ['metrics', 'pred', 'logit', 'truth', 'loss']}
		return data

	def do_calculate(self):

		# Calculating the ORIGINAL findings
		params = {key: getattr(self, key) for key in ['data', 'model', 'device' , 'criterion']}
		self.data = CalculateOriginalFindings.calculate(**params)

		# Adding the ORIGINAL findings to the graph nodes
		self.data.Hierarchy_cls.update_graph( findings_original=self.data.ORIGINAL.results )

		# Saving the results
		LoadSaveFindings(self.config, self.save_path_full).save( self.data.ORIGINAL.results )

	@classmethod
	def get_updated_data(cls, config, data, model):
		""" Getting the ORIGINAL predication probabilities """

		# Initializing the class
		OG = cls(data=data, model=model, config=config, criterion=torch.nn.BCELoss(reduction='none'))

		if config.do_findings_original == 'calculate':
			OG.do_calculate()
		else:
			OG.data.ORIGINAL.results = LoadSaveFindings( config, OG.save_path_full ).load(
			)

			# Adding the ORIGINAL findings to the graph nodes
			OG.data.Hierarchy_cls.update_graph( findings_original=OG.data.ORIGINAL.results )

		return OG.data


class CalculateNewFindings:
	def __init__(self, model: torch.nn.Module, data: Data, hyperparameters: dict, config: argparse.Namespace, methodName: TechniqueNames):

		self.model            = model
		self.hyperparameters  = hyperparameters
		data.initialize_findings(pathologies=self.model.pathologies, experiment_stage=ExperimentStageNames.NEW)
		self.data             = data
		config.methodName     = methodName
		self.config           = config
		self.save_path_full   = f'details/{config.DEFAULT_FINDING_FOLDER_NAME}/{methodName}/findings_new_{data.data_mode}.pkl'

	@staticmethod
	def get_hierarchy_penalty_for_node(graph, node, thresh_technique, parent_condition_mode, methodName, a, b=0) -> np.ndarray:

		def calculate_H(parent_node: str) -> np.ndarray:

			def calculate_raw_weight(pdata) -> pd.Series:
				if   methodName is TechniqueNames.LOGIT_BASED: return pd.Series(a * pdata.data.logit.to_numpy(), index=pdata.data.index)
				elif methodName is TechniqueNames.LOSS_BASED:  return pd.Series(a * pdata.data.loss.to_numpy() + b, index=pdata.data.index)
				else: raise ValueError(' methodName is not supproted')

			def apply_parent_doesnot_exist_condition(hierarchy_penalty: pd.Series, pdata) -> np.ndarray:

				def parent_exist():
					if   parent_condition_mode == 'pred' : return pdata.data.pred >= pdata.metrics.Threshold
					elif parent_condition_mode == 'truth': return pdata.data.truth >= 0.5
					elif parent_condition_mode == 'none' : return pdata.data.truth < 0

				# Setting the hierarchy_penalty to 1 for samples where the parent class exist, because we can not infer any information from those samples.
				hierarchy_penalty[parent_exist()] = 1.0

				# Setting the hierarchy_penalty to 1.0 for samples where we don't have the truth label for parent class.
				hierarchy_penalty[np.isnan( pdata.data.truth )] = 1.0

				return hierarchy_penalty.to_numpy()

			# Getting the parent data
			pdata = Hierarchy.get_findings_for_node(graph=graph, node=parent_node, experimentStage=ExperimentStageNames.ORIGINAL, thresh_technique=thresh_technique)

			# Calculating the initial hierarchy_penalty based on "a" , "b" and "methodName"
			hierarchy_penalty = calculate_raw_weight(pdata)

			# Cleaning up the hierarchy_penalty for the current node: Setting the hierarchy_penalty to 1 for samples where the parent class exist, and Nan if the parent label is Nan
			return apply_parent_doesnot_exist_condition( hierarchy_penalty=hierarchy_penalty, pdata=pdata )

		def set_H_to_be_ineffective() -> np.ndarray:

			ndata = Hierarchy.get_findings_for_node(graph=graph, node=node, thresh_technique=thresh_technique, experimentStage=ExperimentStageNames.ORIGINAL)

			if   methodName is TechniqueNames.LOGIT_BASED: return np.zeros(len(ndata.data.index))
			elif methodName is TechniqueNames.LOSS_BASED:  return np.ones(len(ndata.data.index))

			raise ValueError(' methodName is not supproted')

		# Get the parent node of the current node. We assume that each node can only have one parent to aviod complications in theoretical calculations.
		parent_node = Hierarchy.get_parent_node(graph=graph, node=node)

		# Calculating the hierarchy_penalty for the current node
		return calculate_H(parent_node) if parent_node else set_H_to_be_ineffective()

	@staticmethod
	def do_approach(config, w, ndata):  # type: (argparse.Namespace, pd.Series, Hierarchy.OUTPUT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

		def do_approach2_per_node():

			def calculate_loss_gradient(p, y):  # type: (np.ndarray, np.ndarray) -> pd.Series
				return -y / (p + 1e-7) + (1 - y) / (1 - p + 1e-7)

			def update_pred(l_new, l_gradient):  # type: (pd.Series, pd.Series) -> pd.Series
				p_new = np.exp( -l_new )
				condition = l_gradient >= 0
				p_new[condition] = 1 - p_new[condition]
				return p_new

			# Measuring the new loss values
			loss_new = w.mul( ndata.data.loss.values, axis=0 )

			# Calculating the loss gradient for the old results to find the direction of changes for the new predicted probabilities
			loss_gradient = calculate_loss_gradient( p=ndata.data.pred.to_numpy(), y=ndata.data.truth.to_numpy() )

			# Calculating the new predicted probability
			pred_new = update_pred( l_new=loss_new, l_gradient=loss_gradient )

			return pred_new.to_numpy(), loss_new.to_numpy()

		def do_approach1_per_node():

			# Measuring the new loss values
			logit_new = w.add( ndata.data.logit, axis=0 )

			# Calculating the new predicted probability
			pred_new = 1 / (1 + np.exp( -logit_new ))

			return pred_new.to_numpy(), logit_new.to_numpy()

		if  config.methodName is TechniqueNames.LOGIT_BASED:
			pred_new, logit_new = do_approach1_per_node()
			loss_new = np.ones(pred_new.shape) * np.nan

		elif config.methodName is TechniqueNames.LOSS_BASED:
			pred_new, loss_new  = do_approach2_per_node()
			logit_new = np.ones(pred_new.shape) * np.nan
		else:
			raise ValueError(' methodName is not supported')

		return pred_new, logit_new, loss_new

	@staticmethod
	def calculate_per_node(node: str, data: Data, config: argparse.Namespace, hyperparameters: Dict[ThreshTechList, pd.DataFrame], thresh_technique: ThreshTechList) -> Data:

		x = thresh_technique
		graph = data.Hierarchy_cls.graph

		def initialization():
			# Adding the truth values
			# data.NEW.truth[node] = data.ORIGINAL.truth[node]

			# fixing_dataframes_indices_columns
			index   = data.ORIGINAL.truth.index
			# columns = data.ORIGINAL.truth.columns

			data.NEW.pred              [x,node] = pd.Series(index=index)
			data.NEW.logit             [x,node] = pd.Series(index=index)
			data.NEW.loss              [x,node] = pd.Series(index=index)
			data.NEW.hierarchy_penalty [x,node] = pd.Series(index=index)
		initialization()

		# Getting the hierarchy_penalty for the node
		a, b = hyperparameters[x].loc['a', node], hyperparameters[x].loc['b',node]
		data.NEW.hierarchy_penalty[x, node] = CalculateNewFindings.get_hierarchy_penalty_for_node( graph=graph, a=a, b=b, node=node, thresh_technique=x, parent_condition_mode=config.parent_condition_mode, methodName=config.methodName )

		# Getting node data
		ndata: NodeData = Hierarchy.get_findings_for_node(graph=graph, node=node, thresh_technique=x, experimentStage=ExperimentStageNames.ORIGINAL)

		data.NEW.pred[x, node], data.NEW.logit[x, node], data.NEW.loss[x, node] = CalculateNewFindings.do_approach( config=config, w=data.NEW.hierarchy_penalty[x][node], ndata=ndata )

		data.NEW = TaxonomyXRV.calculating_threshold_and_metrics_per_node(node=node, findings=data.NEW, thresh_technique=x)

		return data

	@staticmethod
	def calculate(data, config, hyperparameters):  # type: (Data, argparse.Namespace, Dict[str, pd.DataFrame]) -> Data

		def initialization():

			# Adding the truth values
			data.NEW.truth = data.ORIGINAL.truth

			# fixing_dataframes_indices_columns
			pathologies = data.ORIGINAL.truth.columns
			columns = pd.MultiIndex.from_product([ThreshTechList, pathologies], names=['thresh_technique', 'pathologies'])
			index   = data.ORIGINAL.truth.index

			# I'm trying to see if I can remove this function, and make the change directly when intializing the Data
			data.NEW.pred              = pd.DataFrame( index=index, columns=columns )
			data.NEW.logit             = pd.DataFrame( index=index, columns=columns )
			data.NEW.loss              = pd.DataFrame( index=index, columns=columns )
			data.NEW.hierarchy_penalty = pd.DataFrame( index=index, columns=columns )

		initialization()
		for x in ThreshTechList:
			for node in data.labels.nodes.non_null:
				data = CalculateNewFindings.calculate_per_node( node=node, data=data, config=config, hyperparameters=hyperparameters, thresh_technique=x )

		data.NEW.results = {key: getattr( data.NEW, key ) for key in ['metrics', 'pred', 'logit', 'truth', TechniqueNames.LOSS_BASED.name, 'hierarchy_penalty']}

		return data

	def do_calculate(self):

		# Calculating the new findings
		params = {key: getattr(self, key) for key in ['data', 'config', 'hyperparameters']}
		self.data = CalculateNewFindings.calculate(**params)

		# Adding the ORIGINAL findings to the graph nodes
		self.data.Hierarchy_cls.update_graph( findings_new=self.data.NEW.results )

		# Saving the new findings
		LoadSaveFindings(self.config, self.save_path_full).save( self.data.NEW.results )

	@classmethod
	def get_updated_data(cls, model: torch.nn.Module, data: Data, hyperparameters: dict, config: argparse.Namespace, methodName: TechniqueNames) -> Data:

		# Initializing the class
		NEW = cls(model=model, data=data, hyperparameters=hyperparameters, config=config, methodName=methodName)

		if config.do_findings_new == 'calculate':
			NEW.do_calculate()
		else:
			NEW.data.NEW.results = LoadSaveFindings( NEW.config, NEW.save_path_full ).load(
			)

			# Adding the ORIGINAL findings to the graph nodes
			NEW.data.Hierarchy_cls.update_graph( findings_new=NEW.data.NEW.results )

		return NEW.data


class HyperParameterTuning:

	def __init__(self, config, data , model):  # type: (argparse.Namespace, Data, torch.nn.Module) -> None

		self.config      = config
		self.data 		 = data
		self.model       = model
		self.save_path_full = f'details/{config.DEFAULT_FINDING_FOLDER_NAME}/{config.methodName}/hyperparameters.pkl'
		self.hyperparameters = None

		# Initializing the data.NEW
		if data.NEW is None:
			data.initialize_findings(pathologies=model.pathologies, experiment_stage=ExperimentStageNames.NEW)

	def initial_hyperparameters(self, a=0.0 , b=1.0):  # type: (float, float) -> Dict[str, pd.DataFrame]
		return {th: pd.DataFrame( {n:dict(a=a,b=b) for n in self.model.pathologies} ) for th in ThreshTechList}

	@staticmethod
	def calculate_per_node(data: Data, config: argparse.Namespace, hyperparameters: Dict[str, pd.DataFrame], node: str, thresh_technique: ThreshTechList=ThreshTechList.DEFAULT) -> List[float]:

		def objective_function(args: Dict[str, float], hp_in) -> float:

			# Updating the hyperparameters for the current node and thresholding technique
			hp_in[thresh_technique][node] = [args['a'], args['b']]

			data2: Data = CalculateNewFindings.calculate_per_node( node=node, data=data, config=config, hyperparameters=hp_in, thresh_technique=thresh_technique)

			# Returning the error
			return 1 - data2.NEW.metrics[thresh_technique, node][config.optimization_metric.upper()]

		# Run the optimization
		best = fmin(
					fn        = lambda args : objective_function( args=args, hp_in=hyperparameters),
					space     = dict(a=hp.uniform('a', -1, 1), b=hp.uniform('b', -4, 4)) ,	# Search space for the variables
					algo      = tpe.suggest ,     # Optimization algorithm (Tree-structured Parzen Estimator)
					max_evals = config.max_evals , # Maximum number of evaluations
					verbose   = True ,	          # Verbosity level (2 for detailed information)
					)

		return [ best['a'] , best['b'] ]

	@staticmethod
	def calculate(initial_hp, config, data):  # type: (Dict[str, pd.DataFrame], argparse.Namespace, Data) -> Dict[str, pd.DataFrame]

		def extended_calculate_per_node(nt: List[str]) -> List[Union[str, float]]:
			hyperparameters_updated = HyperParameterTuning.calculate_per_node( node=nt[0], thresh_technique=nt[1],data=data, config=config, hyperparameters=deepcopy(initial_hp))
			return [nt[0], nt[1]] + hyperparameters_updated # type: ignore

		def update_hyperparameters(results_in):  # type: ( List[Tuple[str, str, List[float]]] ) -> Dict[str, pd.DataFrame]
			# Creating a copy of initial hyperparameters
			hp_in = initial_hp.copy()

			# Updating hyperparameters with new findings
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

	def do_calculate(self):

		if self.config.do_hyperparameters == 'DEFAULT':
			self.hyperparameters = self.initial_hyperparameters()
		else:
			self.hyperparameters = HyperParameterTuning.calculate(initial_hp=deepcopy(self.initial_hyperparameters()), config=self.config, data=self.data)

		# Adding the ORIGINAL findings to the graph nodes
		self.data.Hierarchy_cls.update_graph(hyperparameters=self.hyperparameters)

		# Saving the new findings
		LoadSaveFindings(self.config, self.save_path_full).save(self.hyperparameters)

	@classmethod
	def get_updated_data(cls, config, data, model):
		""" Getting the hyperparameters for the proposed techniques """

		# Initializing the class
		HP = cls(config=config, data=data , model=model)

		if config.do_hyperparameters in ('DEFAULT' ,'calculate'):
			HP.do_calculate()
		else:
			HP.hyperparameters = LoadSaveFindings(config, HP.save_path_full).load()

			# Adding the ORIGINAL findings to the graph nodes
			HP.data.Hierarchy_cls.update_graph(hyperparameters=HP.hyperparameters)

		return HP.hyperparameters


def deserialize_object(state, cls):
	obj = cls.__new__(cls)
	for attr, value in state.items():
		if isinstance(value, dict) and 'pred' in value:
			setattr(obj, attr, pd.DataFrame.from_dict(value))
		elif isinstance(value, dict):
			setattr(obj, attr, deserialize_object(value, globals()[attr]))
		else:
			setattr(obj, attr, value)
	return obj


@dataclass
class MetricsAllTechniques:
	loss            : Metrics
	logit           : Metrics
	auc_acc_f1      : pd.DataFrame
	thresh_technique: str
	datasets_list   : List[str]
	data_mode       : str

	def __post_init__(self):
		self.metrics = self.auc_acc_f1.T[self.list_nodes_impacted].T.astype(float).round(3)

	@property
	def config(self):
		return self.loss.config

	@property
	def list_nodes_impacted(self):
		return self.logit.baseline.list_nodes_impacted


	@staticmethod
	def plot_roc_curves(logit: Metrics, thresh_technique: ThreshTechList, config: argparse.Namespace, list_nodes_impacted: list, save_figure=True, figsize=(15, 15), font_scale=1.8, fontsize=20, labelpad=0):

		def save_plot():
			save_path = config.PATH_LOCAL.joinpath( f'figures/roc_curve_all_datasets/{thresh_technique}/')
			save_path.mkdir(parents=True, exist_ok=True)
			for ft in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(
					f'roc_curve_all_datasets.{ft}'), format=ft, dpi=300)

		# Set up the grid
		def setup_plot():

			# Set a seaborn style for visually appealing plots
			sns.set(font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper', color_codes=True, rc=None)

			# Set up the grid
			n_nodes, n_cols = len(list_nodes_impacted), 3
			n_rows = int(np.ceil(n_nodes / n_cols))

			# Set up the figure and axis
			fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True)  # type: ignore
			axes     = axes.flatten()

			return fig, axes, n_rows, n_cols

		list_parent_nodes = set(Hierarchy.taxonomy_structure.keys()).intersection( set(logit.proposed.pred.columns))

		fig, axes, n_rows, n_cols = setup_plot()

		def plot_per_node(node, idx):

			row_idx = idx // n_cols
			col_idx = idx % n_cols
			ax      = axes[idx]

			# Calculate the ROC curve and AUC
			def get_fpr_tpr_auc(pred_node, truth_node, technique: TechniqueNames, roc_auc):

				def get():
					# mask = ~truth_node.isnull()
					mask = ~np.isnan(truth_node)
					truth_notnull = truth_node[mask].to_numpy()

					if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
						fpr, tpr, _ = sklearn.metrics.roc_curve(truth_notnull, pred_node[mask])
						return fpr, tpr
					return None, None

				fpr, tpr =  get()
				return sns.lineplot(x=fpr, y=tpr, label=f'{technique} AUC = {roc_auc:.2f}', linewidth=2, ax=ax)


			# Plot the ROC curve
			lines, labels = [], []

			truth = logit.proposed.truth
			for methodName in TechniqueNames.members():

				data = getattr(self, methodName.lower())
				technique = TechniqueNames[methodName]

				line = get_fpr_tpr_auc(pred_node=data.pred[node], truth_node=truth[node], technique=technique, roc_auc=data.auc_acc_f1[node][EvaluationMetricNames.AUC.name])
				lines.append(line.lines[-1])
				labels.append(line.get_legend_handles_labels()[1][-1])

			# Customize the plot
			ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
			ax.set_xlabel('False Positive Rate', fontsize=fontsize, labelpad=labelpad) if row_idx == n_rows - 1 else ax.set_xticklabels([])
			ax.set_ylabel('True Positive Rate', fontsize=fontsize, labelpad=labelpad) if col_idx == 0 else ax.set_yticklabels([])
			ax.legend(loc='lower right', fontsize=12)
			ax.set_xlim([0.0, 1.0 ])
			ax.set_ylim([0.0, 1.05])

			leg = ax.legend(lines, labels, loc='lower right', fontsize=fontsize, title=node)
			plt.setp(leg.get_title(),fontsize=fontsize)

			# Set the background color of the plot to grey if the node is a parent node
			if node in list_parent_nodes:
				ax.set_facecolor('xkcd:light grey')

			fig.suptitle('ROC Curves', fontsize=int(1.5*fontsize), fontweight='bold')
			plt.tight_layout()

		def postprocess():
			# Remove any empty plots in the grid
			for empty_idx in range(idx + 1, n_rows * n_cols):
				axes[empty_idx].axis('off')

			plt.tight_layout()

		# Loop through each disease and plot the ROC curve
		for idx, node in enumerate(list_nodes_impacted):
			plot_per_node(node, idx)

		# Postprocess the plot
		postprocess()

		# Save the plot
		if save_figure:
			save_plot()


@dataclass
class MetricsAllTechniqueThresholds:
	DEFAULT: 		  MetricsAllTechniques
	ROC: 			  MetricsAllTechniques
	PRECISION_RECALL: MetricsAllTechniques


class TaxonomyXRV:

	def __init__(self, config: argparse.Namespace, seed: int=10):

		self.hyperparameters = None
		self.config         : argparse.Namespace                  = config
		self.train          : Optional[Data]                      = None
		self.test           : Optional[Data]                      = None
		self.model          : Optional[torch.nn.Module]           = None
		self.dataset         : Optional[xrv.datasets.CheX_Dataset] = None

		methodName = config.methodName or EvaluationMetricNames.LOSS
		self.save_path : str = f'details/{config.DEFAULT_FINDING_FOLDER_NAME}/{methodName}'

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
		hierarchy_penalty = pd.DataFrame(columns=ThreshTechList)
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
		model = LoadModelXRV(config).load()

		# Load the data
		LD = LoadChestXrayDatasets(config=config, pathologies_in_model=model.pathologies)
		LD.load()

		return LD.train, LD.test, model, LD.dataset_full

	@classmethod
	def run_full_experiment(cls, methodName=TechniqueNames.LOSS_BASED, seed=10, **kwargs):

		# Getting the user arguments
		config = get_settings(jupyter=True, methodName=methodName.name, **kwargs)

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
		FE.train = CalculateNewFindings.get_updated_data(data=FE.train, methodName=methodName, **param_dict)
		FE.test  = CalculateNewFindings.get_updated_data(data=FE.test , methodName=methodName, **param_dict)

		# Saving the metrics: AUC, threshold, accuracy
		FE.save_metrics()

		return FE

	@staticmethod
	def loop_run_full_experiment():
		for datasetName in DatasetNames.members():
			for methodName in TechniqueNames:
				TaxonomyXRV.run_full_experiment(methodName=methodName, datasetName=datasetName)

	@classmethod
	def get_merged_data(cls, data_mode=DataModes.TEST, methodName='logit', thresh_technique='DEFAULT', datasets_list=None):  # type: (str, str, str, list) -> Tuple[DataMerged, DataMerged]

		if datasets_list is None:
			datasets_list = DatasetNames

		def get(method: ExperimentStageNames) -> DataMerged:
			data = defaultdict(list)
			for datasetName in datasets_list:
				a1 = cls.run_full_experiment(methodName=methodName, datasetName=datasetName)

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

		def apply_to_approach(methodName: TechniqueNames) -> Metrics:

			baseline, proposed = cls.get_merged_data(data_mode=data_mode, methodName=methodName, thresh_technique=thresh_technique, datasets_list=datasets_list)

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
			# auc_acc_f1_baseline = pd.DataFrame(columns=baseline.pred.columns, index=EvaluationMetricNames.members())
			# auc_acc_f1_proposed = pd.DataFrame(columns=baseline.pred.columns, index=EvaluationMetricNames.members())

			for node in baseline.pred.columns:
				get_auc_acc_f1(node, baseline)
				get_auc_acc_f1(node, proposed)
				get_p_value_kappa_cohen_d_bf10(metrics_comparison, node)

			return Metrics(metrics_comparison=metrics_comparison, baseline=baseline, proposed=proposed, config=config, methodName=methodName)

		def get_auc_acc_f1_merged(logit, loss):  # type: (Metrics, Metrics) -> pd.DataFrame
			columns = pd.MultiIndex.from_product([EvaluationMetricNames.members(), TechniqueNames.members()])
			auc_acc_f1 = pd.DataFrame(columns=columns)

			for metric in EvaluationMetricNames.members():
				auc_acc_f1[metric] = pd.DataFrame( dict(baseline=loss.baseline.auc_acc_f1.T[metric],
														loss=loss.proposed.auc_acc_f1.T[metric],
														logit=logit.proposed.auc_acc_f1.T[metric] ))

			return auc_acc_f1

		if config.do_metrics == 'calculate':
			logit 	   = apply_to_approach(TechniqueNames.LOGIT_BASED)
			loss  	   = apply_to_approach(TechniqueNames.LOSS_BASED)
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




