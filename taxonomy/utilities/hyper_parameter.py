from __future__ import annotations, annotations

import concurrent
import contextlib
import multiprocessing
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from hyperopt import fmin, hp, tpe

from taxonomy.utilities.data import Data
from taxonomy.utilities.metric import CalculateMetricsNode, MetricsNode, Metrics
from taxonomy.utilities.params import TechniqueNames, ThreshTechList
from taxonomy.utilities.settings import Settings
from taxonomy.utilities.utils import remove_null_samples

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'


@dataclass
class HyperPrametersNode:
	MULTIPLIER: float = 1.0
	ADDITIVE  : float = 0.0


# TODO: working on this class
@dataclass
class HyperParameters:
	MULTIPLIER: pd.Series = field(default_factory = pd.Series)
	ADDITIVE  : pd.Series = field(default_factory = pd.Series)

	def __post_init__(self):
		pass

	@classmethod
	def initialize(cls, config: Settings=None, classes: Any=None) -> 'HyperParameters':

		logit = config.technique.technique_name == TechniqueNames.LOGIT
		ADDITIVE   = pd.Series( 0.0, index=classes)
		MULTIPLIER = pd.Series( 0.0 if logit else 1.0, index=classes)

		return cls(MULTIPLIER=MULTIPLIER, ADDITIVE=ADDITIVE)


	def objective_function(config: Settings, findings_original: Findings, node: str):

		threshold_technique = config.hyperparameter_tuning.threshold_technique

		def optimizer(args: dict[str, float]) -> float:

			# Updating the hyper_parameters for the current node and thresholding technique
			model_outputs_node = CalculateNewFindings.calculate_for_node( findings_original=findings_original, node=node, hyper_parameters_node=HyperPrametersNode(**args))

			# Calculate the Metrics
			metrics = CalculateMetrics( REMOVE_NULL=False, truth=model_outputs_node.truth, pred=model_outputs_node.pred, threshold_technique=threshold_technique )

			# Returning the error
			return 1 - metrics.AUC


	def calculate_for_node(self, findings_original: Findings, node: str) -> list[float]:

		config: Settings = findings_original.config

		search_space_multiplier = config.hyperparameter_tuning.search_space_multiplier
		search_space_additive   = config.hyperparameter_tuning.search_space_additive

		threshold_technique = config.hyperparameter_tuning.threshold_technique

		findings_original.model_outputs = remove_null_samples(RAISE_ValueError_IF_EMPTY=True, model_outputs=findings_original.model_outputs[node])




		# Run the optimization
		best = fmin(
				fn        = lambda args : objective_function( args=args, hp_in=hyper_parameters),
				space     = dict( MULTIPLIER = hp.uniform('MULTIPLIER', *search_space_multiplier),
								  ADDITIVE   = hp.uniform('ADDITIVE'  , *search_space_additive)),
				algo      = tpe.suggest ,     # Optimization algorithm (Tree-structured Parzen Estimator)
				max_evals = config.max_evals , # Maximum number of evaluations
				verbose   = True ,	          # Verbosity level (2 for detailed information)
				)

		return [ best['a'] , best['b'] ]

	@classmethod
	def calculate(cls, findings_original: Findings) -> 'HyperParameters':

		config        = findings_original.config
		model_outputs = findings_original.model_outputs
		classes       = findings_original.data.nodes.classes

		# Initializing the hyperparameters
		HP = cls.initialize( config=config, classes=classes )

		CNF = CalculateNewFindings( findings_original=findings_original, hyper_parameters=HP)

		for node in classes:
			HP.MULTIPLIER[node], HP.ADDITIVE[node] = HP.calculate_for_node( model_outputs=model_outputs, config=config, node=node)


		return HP

	def __getitem__(self, node) -> HyperPrametersNode:
		return HyperPrametersNode( MULTIPLIER=self.MULTIPLIER[node], ADDITIVE=self.ADDITIVE[node] )

	def __setitem__(self, node, value: HyperPrametersNode):
		assert isinstance( value, HyperPrametersNode ), "Value must be of type HyperPrametersNode"
		self.MULTIPLIER[node], self.ADDITIVE[node] = value.MULTIPLIER, value.ADDITIVE



@dataclass
class HyperParameterTuning:
	findings_original: Findings

	@staticmethod
	def calculate_per_node(data: Data, config: Settings, hyper_parameters: Dict[str, pd.DataFrame], node: str, threshold_technique: ThreshTechList=ThreshTechList.DEFAULT) -> List[float]:

		def objective_function(args: Dict[str, float], hp_in) -> float:

			# Updating the hyper_parameters for the current node and thresholding technique
			hp_in[threshold_technique][node] = [args['a'], args['b']]

			data2: Data = CalculateNewFindings.calculate_per_node( node=node, data=data, config=config, hyper_parameters=hp_in, threshold_technique=threshold_technique)

			# Returning the error
			return 1 - data2.NEW.metrics[threshold_technique, node][config.optimization_metric.upper()]

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
			hyperparameters_updated = HyperParameterTuning.calculate_per_node( node=nt[0], threshold_technique=nt[1],data=data, config=config, hyper_parameters=deepcopy(initial_hp))
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

