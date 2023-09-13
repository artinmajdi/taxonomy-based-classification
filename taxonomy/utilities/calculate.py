from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from taxonomy.utilities.data import Data, Node
from taxonomy.utilities.findings import Findings, ModelOutputs, ModelOutputsNode
from taxonomy.utilities.hyper_parameter import HyperParameters, HyperPrametersNode
from taxonomy.utilities.metric import CalculateMetrics, Metrics
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

			self.findings.metrics = CalculateMetrics(config=self.config, REMOVE_NULL=True).calculate(model_outputs=self.findings.model_outputs)

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


class CalculateNewFindings:
	findings_original: Findings
	hyper_parameters : HyperParameters

	def __post_init__(self):
		self.config = self.findings_original.config
		self.technique_name = self.config.technique.technique_name

		self.findings = Findings( config = self.findings_original.config,
								  data   = self.findings_original.data,
								  model  = self.findings_original.model,
								  experiment_stage = ExperimentStageNames.NEW)@dataclass


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


class CalculateNewFindingsForNode:
	config: Settings
	node: Node
	node_data: ModelOutputsNode
	parent_data: ModelOutputsNode
	hyper_parameters_node : HyperPrametersNode
	THRESHOLD_parent: float = 0.5

	def __post_init__(self):
		self.technique_name      : TechniqueNames         = self.config.technique.technique_name
		self.parent_metric_to_use: ParentMetricToUseNames = self.config.technique.parent_metric_to_use

	def _calculate_penalty_score(self) -> Union[pd.Series, np.ndarray]:

		def parent_exist_samples():

			if self.parent_metric_to_use == ParentMetricToUseNames.NONE:
				return np.ones_like( self.parent_data.truth, dtype=bool )

			elif self.parent_metric_to_use == ParentMetricToUseNames.TRUTH:
				return self.parent_data.truth >= self.THRESHOLD_parent

			elif self.parent_metric_to_use == ParentMetricToUseNames.PRED:
				return self.parent_data.pred >= self.THRESHOLD_parent

			else:
				raise NotImplementedError(f"ParentMetricToUseNames {self.parent_metric_to_use} not implemented")

		# The hyper_parameters for the current node
		MULTIPLIER: float = self.hyper_parameters_node.MULTIPLIER
		ADDITIVE  : float = self.hyper_parameters_node.ADDITIVE

		# Calculating the initial hierarchy_penalty based on "a", "b" and "technique_name"
		if self.technique_name == TechniqueNames.LOGIT:
			penalty_score = MULTIPLIER * self.parent_data.logit

		elif self.technique_name == TechniqueNames.LOSS:
			penalty_score = MULTIPLIER * self.parent_data.loss + ADDITIVE

		else:
			raise NotImplementedError(f"Technique {self.technique_name} not implemented")

		# Setting to '1' for samples where parent class exist, because we cannot infer any information from those samples.
		penalty_score[parent_exist_samples()] = 1.0

		# Setting the hierarchy_penalty to 1.0 for samples where we don't have the truth label for parent class.
		penalty_score[self.parent_data.truth.isnull()] = 1.0

		return penalty_score

	def _apply_penalty_score_to_node_data(self) -> ModelOutputsNode:

		penalty_score = self._calculate_penalty_score()

		if self.technique_name == TechniqueNames.LOSS:
			# Measuring the new loss values
			self.node_data.loss = penalty_score * self.node_data.loss

			# Calculating the loss gradient to find the direction of changes
			loss_gradient = -self.node_data.truth / (self.node_data.pred + 1e-7) + (1 - self.node_data.truth) / (1 - self.node_data.pred + 1e-7)

			# Calculating the new predicted probability
			self.node_data.pred = np.exp( -self.node_data.loss )

			condition = loss_gradient >= 0
			self.node_data.pred[condition] = 1 - self.node_data.pred[condition]

		elif self.technique_name == TechniqueNames.LOGIT:
			# Measuring the new loss values
			self.node_data.logit = penalty_score + self.node_data.logit

			# Calculating the new predicted probability
			self.node_data.pred = 1 / (1 + np.exp( -self.node_data.logit ))

		else:
			raise NotImplementedError(f"Technique {self.technique_name} not implemented")

		return self.node_data

	def calculate(self) -> ModelOutputsNode:

		if self.node.parent is None:
			return self.node_data

		return self._apply_penalty_score_to_node_data()

