from __future__ import annotations, annotations

from dataclasses import dataclass, field
from typing import Any, Union, TYPE_CHECKING

import pandas as pd
from hyperopt import fmin, hp, tpe

from taxonomy.utilities.params import TechniqueNames
from taxonomy.utilities.utils import node_type_checker

if TYPE_CHECKING:
	from taxonomy.utilities.data import Node
	from taxonomy.utilities.findings import Findings
	from taxonomy.utilities.settings import Settings


@dataclass
class HyperPrametersNode:
	MULTIPLIER: float = 1.0
	ADDITIVE  : float = 0.0


@dataclass
class HyperParameters:
	MULTIPLIER: pd.Series = field(default_factory = pd.Series)
	ADDITIVE  : pd.Series = field(default_factory = pd.Series)

	@classmethod
	def initialize(cls, config: 'Settings'=None, classes: Any=None) -> 'HyperParameters':

		logit      = config.technique.technique_name == TechniqueNames.LOGIT
		MULTIPLIER = pd.Series( 0.0 if logit else 1.0, index=classes)
		ADDITIVE   = pd.Series( 0.0, index=classes)

		return cls(MULTIPLIER=MULTIPLIER, ADDITIVE=ADDITIVE)


	@staticmethod
	def objective_function(findings_original: 'Findings', node: 'Node'):

		from taxonomy.utilities.calculate import UpdateModelOutputs_wrt_Hyperparameters_Node

		params = {  'config'		  : findings_original.config,
					'node'  		  : node,
					'data_node'       : findings_original.model_outputs[node],
					'data_parent'     : findings_original.model_outputs[node.parent],
					'THRESHOLD_node'  : findings_original.metrics.THRESHOLD[node],
					'THRESHOLD_parent': findings_original.metrics.THRESHOLD[node.parent]}

		CNF = UpdateModelOutputs_wrt_Hyperparameters_Node( **params )

		def optimizer(args: dict[str, float]) -> float:
			return 1 - CNF.calculate(args).optimization_metric_value
		return optimizer


	def calculate_for_node(self, findings_original: 'Findings', node: 'Node') -> HyperPrametersNode:

		config: 'Settings' = findings_original.config

		search_space_multiplier = config.hyperparameter_tuning.search_space_multiplier
		search_space_additive   = config.hyperparameter_tuning.search_space_additive

		space = dict(MULTIPLIER = hp.uniform('MULTIPLIER', *search_space_multiplier),
					 ADDITIVE   = hp.uniform('ADDITIVE'  , *search_space_additive))

		objective_function = self.objective_function(findings_original=findings_original, node=node)

		# Run the optimization
		best = fmin( fn        = objective_function,
					 space     = space,
					 algo      = tpe.suggest ,
					 max_evals = config.max_evals ,
					 verbose   = True)

		return HyperPrametersNode(**best)


	@classmethod
	def calculate(cls, findings_original: 'Findings') -> 'HyperParameters':

		# Initializing the hyperparameters
		HP = cls.initialize( config  = findings_original.config,
							 classes = findings_original.data.taxonomy_info.classes )

		# Calculating hyperparameters for all nodes
		for node in findings_original.data.nodes:
			HP[node] = HP.calculate_for_node( findings_original=findings_original, node=node)

		return HP


	@node_type_checker
	def __getitem__(self, node: Union[Node, str]) -> HyperPrametersNode:
		return HyperPrametersNode( MULTIPLIER=self.MULTIPLIER[node], ADDITIVE=self.ADDITIVE[node] )


	@node_type_checker
	def __setitem__(self, node, value: HyperPrametersNode):
		assert isinstance( value, HyperPrametersNode ), "Value must be of type HyperPrametersNode"
		self.MULTIPLIER[node], self.ADDITIVE[node] = value.MULTIPLIER, value.ADDITIVE


'''
@dataclass
class HyperParameterTuning:
	findings_original: 'Findings'

	@staticmethod
	def calculate(initial_hp, config, data):  # type: (Dict[str, pd.DataFrame], Settings, Data) -> Dict[str, pd.DataFrame]

		def extended_calculate_per_node(nt: List[str]) -> List[Union[str, float]]:
			hyperparameters_updated = HyperParameterTuning.calculate_per_node( node=nt[0], threshold_technique=nt[1],data=data, config=config, hyperparameters=deepcopy(initial_hp))
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

'''
