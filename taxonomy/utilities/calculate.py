from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from taxonomy.utilities.data import Data, Node
from taxonomy.utilities.findings import Findings, ModelOutputs, ModelOutputsNode
from taxonomy.utilities.hyperparameters import HyperParameters, HyperPrametersNode
from taxonomy.utilities.metrics import CalculateMetricsNode, Metrics
from taxonomy.utilities.model import ModelType
from taxonomy.utilities.params import ExperimentStageNames, ParentMetricToUseNames, TechniqueNames
from taxonomy.utilities.settings import Settings

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

			self.findings.metrics = Metrics.calculate(config=self.config, model_outputs=self.findings.model_outputs)

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
	hyperparameters : HyperParameters
	config : Settings = field(init=False)

	def __post_init__(self):
		self.config 		= self.findings_original.config
		self.technique_name = self.findings_original.config.technique.technique_name

		self.findings = Findings( config = self.findings_original.config,
								  data   = self.findings_original.data,
								  model  = self.findings_original.model,
								  experiment_stage = ExperimentStageNames.NEW)

	@classmethod
	def calculate(cls, findings_original: Findings, hyperparameters: HyperParameters) -> 'CalculateNewFindings':

		CNF = cls(findings_original=findings_original, hyperparameters=hyperparameters)

		model_outputs_original = CNF.findings_original.model_outputs
		THRESHOLD_original     = CNF.findings_original.metrics.THRESHOLD
		config = findings_original.config

		# Initializing the model_outputs with empty dataframes
		model_outputs_new = ModelOutputs.initialize( columns = model_outputs_original.truth.columns,
												     index   = model_outputs_original.truth.index)


		for node in CNF.findings_original.data.nodes:
			UMOnode = UpdateModelOutputs_wrt_Hyperparameters_Node( config		 = config,
																   node 		 = node,
																   model_outputs = model_outputs_original,
																   THRESHOLD     = THRESHOLD_original)

			model_outputs_new[node] = UMOnode.calculate(args=hyperparameters[node]).model_outputs_node

		CNF.findings.model_outputs = model_outputs_new
		CNF.findings.update_metrics()

		return CNF


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
class UpdateModelOutputs_wrt_Hyperparameters_Node:
	config               : Settings
	node                 : Node
	model_outputs_node   : ModelOutputsNode
	model_outputs_parent : ModelOutputsNode
	THRESHOLD_node       : float
	THRESHOLD_parent     : float
	hyperparameters_node : HyperPrametersNode = field(default=None, init=False)

	def __new__(cls, config: Settings, node: Node = None, **kwargs):

		params = {
			'data_node'       : kwargs.get('data_node')        or kwargs.get('model_outputs')[node],
			'data_parent'     : kwargs.get('data_parent')      or kwargs.get('model_outputs')[node.parent],
			'THRESHOLD_node'  : kwargs.get('THRESHOLD_node')   or kwargs.get('THRESHOLD')[node],
			'THRESHOLD_parent': kwargs.get('THRESHOLD_parent') or kwargs.get('THRESHOLD')[node.parent]
				}

		return super().__new__(cls, config=config, node=node, **params)

	def __post_init__(self):
		self.technique_name      : TechniqueNames = self.config.technique.technique_name
		self.parent_metric_to_use: ParentMetricToUseNames = self.config.technique.parent_metric_to_use

	def _calculate_penalty_score(self) -> Union[pd.Series, np.ndarray]:

		def parent_exist_samples():

			if self.parent_metric_to_use == ParentMetricToUseNames.NONE:
				return np.ones_like( self.model_outputs_parent.truth, dtype=bool )

			elif self.parent_metric_to_use == ParentMetricToUseNames.TRUTH:
				return self.model_outputs_parent.truth >= self.THRESHOLD_parent

			elif self.parent_metric_to_use == ParentMetricToUseNames.PRED:
				return self.model_outputs_parent.pred >= self.THRESHOLD_parent

			else:
				raise NotImplementedError(f"ParentMetricToUseNames {self.parent_metric_to_use} not implemented")

		# The hyperparameters for the current node
		MULTIPLIER: float = self.hyperparameters_node.MULTIPLIER
		ADDITIVE  : float = self.hyperparameters_node.ADDITIVE

		# Calculating the initial hierarchy_penalty based on "a", "b" and "technique_name"
		if self.technique_name == TechniqueNames.LOGIT:
			penalty_score = MULTIPLIER * self.model_outputs_parent.logit

		elif self.technique_name == TechniqueNames.LOSS:
			penalty_score = MULTIPLIER * self.model_outputs_parent.loss + ADDITIVE

		else:
			raise NotImplementedError(f"Technique {self.technique_name} not implemented")

		# Setting to '1' for samples where parent class exist, because we cannot infer any information from those samples.
		penalty_score[parent_exist_samples()] = 1.0

		# Setting the hierarchy_penalty to 1.0 for samples where we don't have the truth label for parent class.
		penalty_score[self.model_outputs_parent.truth.isnull()] = 1.0

		return penalty_score

	def apply_penalty(self) -> ModelOutputsNode:

		penalty_score = self._calculate_penalty_score()

		if self.technique_name == TechniqueNames.LOSS:
			# Measuring the new loss values
			self.model_outputs_node.loss = penalty_score * self.model_outputs_node.loss

			# Calculating the loss gradient to find the direction of changes
			loss_gradient = -self.model_outputs_node.truth / (self.model_outputs_node.pred + 1e-7) + (1 - self.model_outputs_node.truth) / (1 - self.model_outputs_node.pred + 1e-7)

			# Calculating the new predicted probability
			self.model_outputs_node.pred = np.exp( -self.model_outputs_node.loss )

			condition = loss_gradient >= 0
			self.model_outputs_node.pred[condition] = 1 - self.model_outputs_node.pred[condition]

		elif self.technique_name == TechniqueNames.LOGIT:
			# Measuring the new loss values
			self.model_outputs_node.logit = penalty_score + self.model_outputs_node.logit

			# Calculating the new predicted probability
			self.model_outputs_node.pred = 1 / (1 + np.exp( -self.model_outputs_node.logit ))

		else:
			raise NotImplementedError(f"Technique {self.technique_name} not implemented")

		return self.model_outputs_node

	def calculate(self, args: Union[HyperPrametersNode, dict]) -> 'UpdateModelOutputs_wrt_Hyperparameters_Node':

		self.hyperparameters_node = args if isinstance( args, HyperPrametersNode ) else HyperPrametersNode( **args )

		if self.node.parent is not None:
			self.apply_penalty()

		return self

	@property
	def optimization_metric_value(self) -> float:
		CMN = CalculateMetricsNode( node 			   = self.node,
									model_outputs_node = self.model_outputs_node,
									config             = self.config,
									THRESHOLD          = self.THRESHOLD_node )

		return getattr(CMN, self.config.hyperparameter_tuning.optimization_metric.name)

