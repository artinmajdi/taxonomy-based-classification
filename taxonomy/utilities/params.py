from __future__ import annotations

import argparse
import enum
import itertools
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Set, Tuple, Union, Dict

import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader as torch_DataLoader
import torchxrayvision as xrv


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


@dataclass
class Nodes:
	CLASSES          : Set[str]
	NON_NULL         : Set[str] 		   = field(default_factory = set)
	IMPACTED         : Set[str] 		   = field(default_factory = set)
	TAXONOMY         : Dict[str, Set[str]] = field(default_factory = dict)
	EXIST_IN_TAXONOMY: Set[str] 	       = field(default_factory = set)

	def __post_init__(self):
		self.TAXONOMY          = self._taxonomy()
		self.EXIST_IN_TAXONOMY = self._exist_in_taxonomy()

	@staticmethod
	def _default_taxonomy() -> Dict[str, Set[str]]:
		return {'Lung Opacity': {'Pneumonia', 'Atelectasis', 'Consolidation', 'Lung Lesion', 'Edema', 'Infiltration'}, 'Enlarged Cardiomediastinum': {'Cardiomegaly'}}

	def _taxonomy(self):  # sourcery skip: dict-comprehension
		UT = {}
		for parent in self._default_taxonomy():
			if parent in self.CLASSES:
				UT[parent] = self._default_taxonomy()[parent].intersection(self.CLASSES)

		return UT or None

	def _exist_in_taxonomy(self) -> Set[str]:

		# Adding the parent classes
		eit = set(self.TAXONOMY.keys())

		# Adding the children classes
		for value in self.TAXONOMY.values():
			eit.update(value)

		return eit

	@property
	def node_thresh_tuple(self):
		return list(itertools.product(self.IMPACTED, ThreshTechList))

	def get_children_of(self, parent: str) -> Set[str]:
		return self.TAXONOMY.get(parent, set())

	def get_parent_of(self, child: str) -> str | None:
		return next((parent for parent in self.TAXONOMY if child in self.TAXONOMY[parent]), None)


@dataclass
class NodeData:
	G    : nx.DiGraph = field(default_factory = lambda: None)
	nodes: Nodes      = field(default_factory = lambda: None)

	def __post_init__(self):
		self.G = nx.DiGraph(self.nodes.TAXONOMY)
		self.G.add_nodes_from(self.nodes.CLASSES)

	def add_hyperparameters_to_node(self, parent_node: str, child_node: str, hyperparameter: dict[HyperparameterNames, float]):
		for hp_name in hyperparameter:
			self.G.edges[parent_node, child_node][hp_name] = hyperparameter[hp_name]

	def get_hyperparameters_of_node(self, parent_node: str, child_node: str) -> dict[HyperparameterNames, float]:
		return self.G.edges[parent_node, child_node]

	def add_hierarchy_penalty(self, node: str, penalty: float):
		self.G.nodes[node]['hierarchy_penalty'] = penalty

	def get_hierarchy_penalty(self, node: str) -> float:
		return self.G.nodes[node].get('hierarchy_penalty', None)

	def show(self, package='networkx'):

		import plotly.graph_objs as go
		import seaborn as sns

		def plot_networkx_digraph(G):
			pos = nx.spring_layout(G)
			edge_x = []
			edge_y = []
			for edge in G.edges():
				x0, y0 = pos[edge[0]]
				x1, y1 = pos[edge[1]]
				edge_x.append(x0)
				edge_x.append(x1)
				edge_x.append(None)
				edge_y.append(y0)
				edge_y.append(y1)
				edge_y.append(None)

			edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none',
									mode='lines')

			node_x = [pos[node][0] for node in G.nodes()]
			node_y = [pos[node][1] for node in G.nodes()]

			colorbar = dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')
			marker = dict(showscale=True, colorscale='Viridis', reversescale=True, color=[], size=10, line_width=2,
						  colorbar=colorbar)

			node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=marker)

			node_adjacency = []
			node_text = []
			for node, adjacency in enumerate(G.adjacency()):
				node_adjacency.append(len(adjacency[1]))
				node_text.append(f'{adjacency[0]} - # of connections: {len(adjacency[1])}')

			node_trace.marker.color = node_adjacency
			node_trace.text = node_text

			# Add node labels
			annotations = []
			for node in G.nodes():
				x, y = pos[node]
				annotations.append(
						go.layout.Annotation(text=str(node), x=x, y=y, showarrow=False, font=dict(size=10, color='black')))

			layout = go.Layout(title='Networkx DiGraph',
							   showlegend = False,
							   hovermode  = 'closest',
							   margin     = dict(b    = 20, l = 5, r = 5, t = 40),
							   xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
							   yaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
							   annotations = annotations
							   )
			fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
			fig.show()

		if package == 'networkx':
			sns.set_style("whitegrid")
			nx.draw(self.G, with_labels=True)

		elif package == 'plotly':
			plot_networkx_digraph(self.G)


@dataclass
class Labels:
	LABEL_SET: Union[np.ndarray, pd.DataFrame]
	CLASSES  : list[str]
	nodes	 : Nodes = field(default_factory=lambda: None)

	def __post_init__(self):
		self.nodes: Nodes = Nodes(CLASSES=set(self.CLASSES))
		self.LABEL_SET = pd.DataFrame(self.LABEL_SET, columns=self.CLASSES)

		if self.LABEL_SET.size > 0:
			self.nodes.NON_NULL    = self.LABEL_SET.columns[self.LABEL_SET.count() > 0].to_list()
			self.nodes.IMPACTED    = [x for x in self.nodes.NON_NULL if x in self.nodes.EXIST_IN_TAXONOMY]


@members
class DataModes(enum.Enum):
	TRAIN = 'train'
	TEST  = 'test'
	ALL   = 'all'


@dataclass
class Data:
	dataset    : xrv.datasets.Dataset
	data_loader: torch_DataLoader
	labels     : Labels    = field(default_factory = lambda: None)
	dataMode   : DataModes = field(default_factory = lambda: None)

	def __post_init__(self):
		self.labels = Labels(LABEL_SET=self.dataset.labels, CLASSES=self.dataset.pathologies)

	
@members
class TechniqueNames(enum.Enum):
	BASELINE = 'baseline'
	LOGIT    = 'logit_based'
	LOSS     = 'loss_based'


@members
class ModelWeightNames(enum.Enum):
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
	THRESHOLD = 'threshold'


@dataclass
class Metrics:
	ACC: pd.DataFrame = field(default_factory = lambda: None)
	AUC: pd.DataFrame = field(default_factory = lambda: None)
	F1 : pd.DataFrame = field(default_factory = lambda: None)
	THRESHOLD: dict[ThreshTechList, pd.DataFrame] = field(default_factory = lambda: None)


@members
class FindingNames(enum.Enum):
	GROUND_TRUTH = 'ground_truth'
	LOSS_VALUES  = 'loss_values'
	LOGIT_VALUES = 'logit_values'
	PRED_PROBS   = 'pred_probs'


@dataclass
class Findings:
	data           : Data                 = field(default_factory = lambda: None)
	ground_truth   : pd.DataFrame         = field(default_factory = lambda: None)
	loss_values    : pd.DataFrame         = field(default_factory = lambda: None)
	logit_values   : pd.DataFrame         = field(default_factory = lambda: None)
	pred_probs     : pd.DataFrame         = field(default_factory = lambda: None)
	metrics        : Metrics              = field(default_factory = lambda: None)
	nodeData	   : NodeData             = field(default_factory = lambda: None)
	techniqueName  : TechniqueNames       = field(default_factory = lambda: None)
	experimentStage: ExperimentStageNames = field(default_factory = lambda: None)

@members
class HyperparameterNames(enum.Enum):
	A = 'a'
	B = 'b'


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

			args.methodName  = TechniqueNames[args.methodName.upper()]
			args.datasetName = DatasetNames[args.datasetName.upper()]
			args.modelName   = ModelWeightNames[args.modelName.upper()]

			args.DEFAULT_FINDING_FOLDER_NAME = f'{args.datasetName}-{args.modelName}'

		return args

	# Updating the config file
	return  get_config(args=parse_args())


if __name__ == '__main__':
	print(ExperimentStageNames.members())

	