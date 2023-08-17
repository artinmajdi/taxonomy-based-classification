import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from torch.utils.data import DataLoader as torch_DataLoader

import torchxrayvision as xrv
from taxonomy.utilities.params import DataModes, ThreshTechList, DatasetNames, ExperimentStageNames, \
	EvaluationMetricNames, ModelFindingNames, MethodNames, NodeData


class Data:
	data_loader: torch_DataLoader = None
	list_datasets = DatasetNames
	
	def __init__(self, data_mode: Union[str, DataModes]):
		self.data_mode = data_mode.value if isinstance(data_mode, DataModes) else data_mode
		
		self._d_data: Optional[xrv.datasets.CheX_Dataset] = None
		self.data_loader: Optional[torch_DataLoader] = None
		self.Hierarchy_cls: Optional[Hierarchy] = None
		self.ORIGINAL: Optional[Findings] = None
		self.NEW: Optional[Findings] = None
		self.labels: Optional[Labels] = None
		self.list_findings_names: list = []
	
	@property
	def d_data(self):
		return self._d_data
	
	@d_data.setter
	def d_data(self, value):
		if value is not None:
			self._d_data = value
			
			# Adding the initial Graph
			self.Hierarchy_cls = Hierarchy(value.pathologies)
			self.labels = Labels(d_data=value, Hierarchy_cls=self.Hierarchy_cls)
	
	def initialize_findings(self, pathologies: list[str], experiment_stage: Union[str, ExperimentStageNames]):
		setattr(self, experiment_stage, Findings(experiment_stage=experiment_stage, pathologies=pathologies))
		
		self.list_findings_names.append(experiment_stage)


class Labels:
	def __init__(self, d_data, Hierarchy_cls):
		self.d_data = d_data
		self.labels = pd.DataFrame(d_data.labels, columns=d_data.pathologies) if d_data else pd.DataFrame()
		self.nodes = self._get_nodes(Hierarchy_cls)
		self.totals = pd.DataFrame(self.d_data.totals()) if d_data else pd.DataFrame()
	
	def _get_nodes(self, Hierarchy_cls):
		
		@dataclass
		class Nodes:
			unique: List[str] = field(default_factory=list)
			parents: List[str] = field(default_factory=list)
			exist_in_taxonomy: List[str] = field(default_factory=list)
			not_null: List[str] = field(default_factory=list)
			impacted: List[str] = field(default_factory=list)
			node_thresh_tuple: List[str] = field(default_factory=list)
		
		nodes = Nodes()
		nodes.unique = self.totals.index.to_list()
		nodes.parents = set(Hierarchy.taxonomy_structure.keys())
		
		if Hierarchy_cls:
			nodes.exist_in_taxonomy = Hierarchy_cls.list_nodes_exist_in_taxonomy
		
		if self.labels.size > 0:
			nodes.not_null = self.labels.columns[self.labels.count() > 0].to_list()
			nodes.impacted = [x for x in nodes.not_null if x in nodes.exist_in_taxonomy]
			nodes.node_thresh_tuple = list(itertools.product(nodes.impacted, ThreshTechList))
		
		return nodes


class Findings:
	list_metrics = EvaluationMetricNames.members() + ['Threshold']
	list_arguments = ['metrics'] + ModelFindingNames.members()
	
	def __init__(self, pathologies: List[str], experiment_stage: Union[str, ExperimentStageNames]):
		
		self.experiment_stage = ExperimentStageNames[experiment_stage.upper()] if isinstance(experiment_stage,
		                                                                                     str) else experiment_stage
		
		self.pathologies = pathologies
		
		# Initializing Metrics & Thresholds
		columns = pd.MultiIndex.from_product([ThreshTechList, self.pathologies],
		                                     names=['thresh_technique', 'pathologies'])
		
		self.metrics = pd.DataFrame(columns=columns, index=Findings.list_metrics)
		
		# Initializing Arguments & Findings
		self.truth = pd.DataFrame(columns=pathologies)
		
		if self.experiment_stage == ExperimentStageNames.ORIGINAL:
			self.pred = pd.DataFrame(columns=pathologies)
			self.logit = pd.DataFrame(columns=pathologies)
			self.loss = pd.DataFrame(columns=pathologies)
			self._results = {key: getattr(self, key) for key in Findings.list_arguments}
		
		elif self.experiment_stage == ExperimentStageNames.NEW:
			self.pred = pd.DataFrame(columns=columns)
			self.logit = pd.DataFrame(columns=columns)
			self.loss = pd.DataFrame(columns=columns)
			self.hierarchy_penalty = pd.DataFrame(columns=columns)
			self._results = {key: getattr(self, key) for key in Findings.list_arguments + ['hierarchy_penalty']}
	
	@property
	def results(self):
		return self._results
	
	@results.setter
	def results(self, value):
		self._results = value
		
		if value is not None:
			for key in Findings.list_arguments:
				setattr(self, key, value[key])
			
			if self.experiment_stage == ExperimentStageNames.NEW:
				setattr(self, 'hierarchy_penalty', value['hierarchy_penalty'])


class Hierarchy:
	taxonomy_structure = {
		'Lung Opacity': ['Pneumonia', 'Atelectasis', 'Consolidation', 'Lung Lesion', 'Edema', 'Infiltration'],
		'Enlarged Cardiomediastinum': ['Cardiomegaly']
	}
	
	# 'Infiltration'              : ['Consolidation'],
	# 'Consolidation'             : ['Pneumonia'],
	
	def __init__(self, classes=None):
		
		if classes is None:
			classes = []
		
		self.classes = list(set(classes))
		
		# Creating the taxonomy for the dataset
		self.hierarchy = self.create_hierarchy(self.classes)
		
		# Creating the Graph
		self.G = Hierarchy.create_graph(classes=self.classes, hierarchy=self.hierarchy)
	
	@staticmethod
	def create_graph(classes, hierarchy):
		G = nx.DiGraph(hierarchy)
		G.add_nodes_from(classes)
		return G
	
	def update_graph(self, classes=None, findings_original=None, findings_new=None, hyperparameters=None):
		
		def add_nodes_and_edges(hierarchy=None):
			if classes and len(classes) > 0: self.G.add_nodes_from(classes)
			if hierarchy and len(hierarchy) > 0: self.G.add_edges_from(hierarchy)
		
		def add_hyperparameters_to_graph():
			for parent_node, child_node in self.G.edges:
				self.G.edges[parent_node, child_node]['hyperparameters'] = {x: hyperparameters[x][child_node].copy() for
				                                                            x in ThreshTechList}
		
		def graph_add_findings_original_to_nodes(findings: dict):
			
			# Loop over all classes (aka nodes)
			for n in findings[ModelFindingNames.TRUTH].columns:
				
				data = pd.DataFrame({m.value: findings[m][n] for m in ModelFindingNames})
				
				metrics = {}
				for x in ThreshTechList:
					metrics[x] = findings['metrics'][x, n]
				
				# Adding the findings to the graph node
				self.G.nodes[n][ExperimentStageNames.ORIGINAL] = dict(data=data, metrics=metrics)
		
		def graph_add_findings_new_to_nodes(findings: dict):
			
			# Loop over all classes (aka nodes)
			for n in findings['truth'].columns:
				
				# Merging the pred, truth, and loss into a dataframe
				data, metrics, hierarchy_penalty = {}, {}, pd.DataFrame()
				for x in ThreshTechList:
					data[x] = pd.DataFrame(
						dict(truth=findings['truth'][n], pred=findings['pred'][x, n],
						     loss=findings[MethodNames.LOSS_BASED.name][x, n]))
					metrics[x] = findings['metrics'][x, n]
					hierarchy_penalty[x] = findings['hierarchy_penalty'][x, n].values
				
				# Adding the findings to the graph node
				self.G.nodes[n][ExperimentStageNames.NEW] = dict(data=data, metrics=metrics)
				
				# Updating the graph with the hierarchy_penalty for the current node
				self.G.nodes[n]['hierarchy_penalty'] = hierarchy_penalty
		
		if classes: add_nodes_and_edges()
		if hyperparameters: add_hyperparameters_to_graph()
		if findings_new: graph_add_findings_new_to_nodes(findings_new)
		if findings_original: graph_add_findings_original_to_nodes(findings_original)
	
	def add_hyperparameters_to_node(self, parent, child, a=1.0, b=0.0):  # type: (str, str, float, float) -> None
		self.G.edges[parent, child]['a'] = a
		self.G.edges[parent, child]['b'] = b
	
	def graph_get_taxonomy_hyperparameters(self, parent_node, child_node):  # type: (str, str) -> Tuple[float, float]
		return self.G.edges[parent_node, child_node]['hyperparameters']
	
	@staticmethod
	def get_node_original_results_for_child_and_parent_nodes(G: nx.DiGraph, node: str, thresh_technique: ThreshTechList=ThreshTechList.ROC) -> Tuple[NodeData, NodeData]:
		
		# The predicted probability p, true label y, optimum threshold th, and loss for node class
		child_data = Hierarchy.get_findings_for_node(G=G, node=node, thresh_technique=thresh_technique,
		                                             experimentStage=ExperimentStageNames.ORIGINAL)
		
		# The predicted probability p, true label y, optimum threshold th, and loss for parent class
		parent_node = Hierarchy.get_parent_node(G=G, node=node)
		parent_data = Hierarchy.get_findings_for_node(G=G, node=parent_node, thresh_technique=thresh_technique,
		                                              experimentStage=ExperimentStageNames.ORIGINAL) if parent_node else None
		
		return child_data, parent_data
	
	@staticmethod
	def get_findings_for_node(G: nx.DiGraph, node: str, thresh_technique: ThreshTechList, experimentStage: ExperimentStageNames):
		WR = experimentStage
		TT = thresh_technique
		
		node_data = G.nodes[node]
		
		if WR == ExperimentStageNames.ORIGINAL:
			data    = node_data[WR]['data']
			metrics = node_data[WR]['metrics'][TT]
			hierarchy_penalty = None
			
		elif WR == ExperimentStageNames.NEW:
			data    = node_data[WR]['data'][TT]
			metrics = node_data[WR]['metrics'][TT]
			hierarchy_penalty = node_data['hierarchy_penalty']
			
		else:
			raise ValueError('experimentStage must be either ORIGINAL or NEW')
			
		return NodeData(data=data, metrics=metrics, hierarchy_penalty=hierarchy_penalty)
	
	@staticmethod
	def get_parent_node(G, node):  # type: (nx.DiGraph, str) -> str
		parent_node = nx.dag.ancestors(G, node)
		return list(parent_node)[0] if (len(parent_node) > 0) else None
	
	# @staticmethod
	# def taxonomy_structure(self):
	# 	# 'Infiltration'              : ['Consolidation'],
	# 	# 'Consolidation'             : ['Pneumonia'],
	# 	return {
	# 			'Lung Opacity'              : ['Pneumonia', 'Atelectasis', 'Consolidation', 'Lung Lesion', 'Edema', 'Infiltration'],
	# 			'Enlarged Cardiomediastinum': ['Cardiomegaly'],
	# 			}
	
	@property
	def parent_dict(self):
		p_dict = defaultdict(list)
		for parent_i, its_children in self.hierarchy.items():
			for child in its_children:
				p_dict[child].append(parent_i)
		return p_dict
	
	@property
	def child_dict(self):
		return self.hierarchy
	
	@property
	def list_nodes_exist_in_taxonomy(self):
		LPI = set()
		for key, value in Hierarchy.taxonomy_structure.items():
			LPI.update([key])
			LPI.update(value)
		
		return LPI.intersection(set(self.classes))
	
	@staticmethod
	def create_hierarchy(classes: list):
		
		UT = defaultdict(list)
		
		# Looping through all parent classes in the DEFAULT taxonomy
		for parent in Hierarchy.taxonomy_structure.keys():
			
			# Check if the parent class of our DEFAULT taxonomy exist in the dataset
			if parent in classes:
				for child in Hierarchy.taxonomy_structure[parent]:
					
					# Checks if the child classes of found parent class exist in the dataset
					if child in classes:
						UT[parent].append(child)
		
		return UT or None
	
	@property
	def parent_child_df(self):
		df = pd.DataFrame(columns=['parent', 'child'])
		for p in self.hierarchy.keys():
			for c in self.hierarchy[p]:
				df = df.append({'parent': p, 'child': c}, ignore_index=True)
		return df
	
	def show(self, package='networkx'):
		
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
