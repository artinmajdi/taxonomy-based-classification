import argparse
import itertools
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import torch
import torchvision
from torch.utils.data import DataLoader as torch_DataLoader

import torchxrayvision as xrv
from taxonomy.utilities.model import LoadModelXRV
from taxonomy.utilities.params import DatasetNames, EvaluationMetricNames, ExperimentStageNames, MethodNames, \
	ModelFindingNames, NodeData, ThreshTechList

USE_CUDA = torch.cuda.is_available()


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
		for parent in self._default_taxonomy().keys():
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


@dataclass
class Labels:
	LABEL_SET: Union[np.ndarray, pd.DataFrame]
	CLASSES  : List[str]
	nodes	 : Nodes = field(default_factory=lambda: None)

	def __post_init__(self):
		self.nodes: Nodes = Nodes(CLASSES=set(self.CLASSES))
		self.LABEL_SET = pd.DataFrame(self.LABEL_SET, columns=self.CLASSES)

		if self.LABEL_SET.size > 0:
			self.nodes.NON_NULL    = self.LABEL_SET.columns[self.LABEL_SET.count() > 0].to_list()
			self.nodes.IMPACTED    = [x for x in self.nodes.NON_NULL if x in self.nodes.EXIST_IN_TAXONOMY]


@dataclass
class Data:
	dataset    : xrv.datasets.Dataset
	data_loader: torch_DataLoader
	labels     : Labels = field(default_factory = lambda: None)

	def __post_init__(self):
		self.labels = Labels(LABEL_SET=self.dataset.labels, CLASSES=self.dataset.pathologies)


@dataclass(init=False)
class DataMerged:
	def __init__(self, data):
		self.pred       = self.concat_data(data['pred'])
		self.truth      = self.concat_data(data['truth'])
		self.yhat       = self.concat_data(data['yhat']  , int)
		self.auc_acc_f1 = pd.DataFrame(columns=self.pred.columns, index=EvaluationMetricNames.members())
		self.list_nodes_impacted = [n for n in self.pred.columns if n in set().union(*data['list_nodes_impacted'])]

	@staticmethod
	def concat_data(data, dtype=None):
		df = pd.concat(data, axis=0).reset_index(drop=True)
		return df.astype(dtype) if dtype else df


@dataclass
class Metrics:
	metrics_comparison: pd.DataFrame
	proposed          : DataMerged


@dataclass
class MethodFindings:
	logit_techniques: Metrics
	loss_technique  : Metrics
	baseline		: Metrics
	data: Data


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


@dataclass
class Hierarchy:
	classes: List[str]  = field(default_factory = list)
	G      : nx.DiGraph = field(default_factory = lambda: None)

	def __post_init__(self):
		self.G       = Hierarchy.create_graph(classes = self.classes, hierarchy = self.hierarchy)
		self.classes = list(set(self.classes))

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


@dataclass
class LoadChestXrayDatasets:
	config : argparse.Namespace   = field(default_factory = lambda: None)
	dataset: xrv.datasets.Dataset = field(default_factory = lambda: None)
	train  : Data                 = field(default_factory = lambda: None)
	test   : Data                 = field(default_factory = lambda: None)

	def load_raw_database(self):
		"""
			# RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
				Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
				Annotations of Possible Pneumonia.	Shih, George, Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
				More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
				Challenge site:	https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
				JPG files stored here:	https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9

			# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031
				Dataset website here: https://stanfordmlgroup.github.io/competitions/chexpert/

			# NIH ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
				Dataset release website:
				https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

				Download full size images here:
				https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

				Download resized (224x224) images here:
				https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0

			# PadChest: A large chest thresh_technique-ray image dataset with multi-label annotated reports. https://arxiv.org/abs/1901.07441
				Note that images with null labels (as opposed to normal), and images that cannot
				be properly loaded (listed as 'missing' in the code) are excluded, which makes
				the total number of available images slightly less than the total number of image
				files.

				PadChest: A large chest thresh_technique-ray image dataset with multi-label annotated reports.
				Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá.
				arXiv preprint, 2019. https://arxiv.org/abs/1901.07441
				Dataset website: https://bimcv.cipf.es/bimcv-projects/padchest/
				Download full size images here: https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850
				Download resized (224x224) images here (recropped):	https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797

			# VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
				VinBrain Dataset. Nguyen et al., VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations
				https://arxiv.org/abs/2012.15029
				https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

			# MIMIC-CXR Dataset
				Johnson AE,	MIMIC-CXR: A large publicly available database of labeled chest radiographs.
				arXiv preprint arXiv:1901.07042. 2019 Jan 21.	https://arxiv.org/abs/1901.07042
				Dataset website here:	https://physionet.org/content/mimic-cxr-jpg/2.0.0/

			# OpenI Dataset
				Dina Demner-Fushman, Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American
				Medical Informatics Association, 2016. doi: 10.1093/jamia/ocv080.

				Views have been determined by projection using T-SNE.  To use the T-SNE view rather than the
				view defined by the record, set use_tsne_derived_view to true.
				Dataset website: https://openi.nlm.nih.gov/faq
				Download images: https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d

			# NIH_Google Dataset
				A relabeling of a subset of images from the NIH dataset.  The data tables should
				be applied against an NIH download.  A test and validation split are provided in the
				original.  They are combined here, but one or the other can be used by providing
				the original csv to the csvpath argument.

				Chest Radiograph Interpretation with Deep Learning Models: Assessment with
				Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
				Anna Majkowska. Radiology 2020	https://pubs.rsna.org/doi/10.1148/radiol.2019191293
				NIH data can be downloaded here:	https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
		"""

		imgpath   = self._path_dataset()
		views     = self.config.views
		csvpath   = self._path_csv_files()
		transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size = 224, engine = "cv2")])

		params_config = {
				DatasetNames.NIH       : dict(imgpath = imgpath, views = views),
				DatasetNames.PC        : dict(imgpath = imgpath, views = views),
				DatasetNames.CheXPERT  : dict(imgpath = imgpath, views = views , transform = transform , csvpath = csvpath),
				DatasetNames.MIMIC     : dict(imgpath = imgpath, views = views, transform = transform, csvpath = csvpath, metacsvpath = self._path_meta_data_csv_files() ),
				DatasetNames.Openi     : dict(imgpath = imgpath, views = views , transform = transform),
				DatasetNames.VinBrain  : dict(imgpath = imgpath, views = views , csvpath   = csvpath),
				DatasetNames.RSNA      : dict(imgpath = imgpath, views = views , transform = transform),
				DatasetNames.NIH_Google: dict(imgpath = imgpath, views = views)
				}

		dataset_config = {
				DatasetNames.NIH       : xrv.datasets.NIH_Dataset,
				DatasetNames.PC        : xrv.datasets.PC_Dataset,
				DatasetNames.CheXPERT  : xrv.datasets.CheX_Dataset,
				DatasetNames.MIMIC     : xrv.datasets.MIMIC_Dataset,
				DatasetNames.Openi     : xrv.datasets.Openi_Dataset,
				DatasetNames.VinBrain  : xrv.datasets.VinBrain_Dataset,
				DatasetNames.RSNA      : xrv.datasets.RSNA_Pneumonia_Dataset,
				DatasetNames.NIH_Google: xrv.datasets.NIH_Google_Dataset
				}

		return dataset_config.get(self.config.datasetName)(**params_config.get(self.config.datasetName))

	def relabel_raw_database(self):
		# Adding the PatientID if it doesn't exist
		# if "patientid" not in self.dataset.csv:
		# 	self.dataset.csv["patientid"] = [ f"{self.dataset.__class__.__name__}-{i}" for i in range(len(self.dataset)) ]

		# Filtering the dataset
		# self.dataset.csv = self.dataset.csv[self.dataset.csv['Frontal/Lateral'] == 'Frontal'].reset_index(drop=False)

		# Aligning labels to have the same order as the pathologies' argument.
		xrv.datasets.relabel_dataset( pathologies=LoadModelXRV.model_classes(self.config), dataset=self.dataset, silent=self.config.silent)

	def update_empty_parent_class_based_on_its_children_classes(self):

		labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

		# This will import the child_dict
		child_dict = Hierarchy(classes=self.dataset.pathologies).child_dict

		for parent, children in child_dict.items():

			# Checking if the parent class existed in the original pathologies in the dataset.
			# Will only replace its values if all its labels are NaN
			if labels[parent].value_counts().values.shape[0] == 0:
				if not self.config.silent:
					print(f"Parent class: {parent} is not in the dataset. replacing its true values according to its children presence.")

				# Initializing the parent label to 0
				labels[parent] = 0

				# If at-least one of the children has a label of 1, then the parent label is 1
				labels[parent][ labels[children].sum(axis=1) > 0 ] = 1

		self.dataset.labels = labels.values

	def load(self):

		def train_test_split():
			labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

			idx_train = labels.sample(frac=self.config.train_test_ratio).index
			d_train = xrv.datasets.SubsetDataset(self.dataset, idxs=idx_train)

			idx_test = labels.drop(idx_train).index
			d_test = xrv.datasets.SubsetDataset(self.dataset, idxs=idx_test)

			return d_train, d_test

		def selecting_non_null_samples():
			"""  Selecting non-null samples for impacted pathologies  """

			dataset = self.dataset
			columns = self.dataset.pathologies
			labels  = pd.DataFrame( dataset.labels , columns=columns)

			# This will import the child_dict
			child_dict = Hierarchy(classes=self.dataset.pathologies).child_dict

			# Looping through all parent classes in the taxonomy
			for parent in child_dict.keys():

				# Extracting the samples with a non-null value for the parent truth label
				labels  = labels[ ~labels[parent].isna() ]
				dataset = xrv.datasets.SubsetDataset(dataset, idxs=labels.index)
				labels  = pd.DataFrame( dataset.labels , columns=columns)

				# Extracting the samples, where for each parent, at least one of their children has a non-null truth label
				labels  = labels[ (~labels[ child_dict[parent] ].isna()).sum(axis=1) > 0 ]
				dataset = xrv.datasets.SubsetDataset(dataset, idxs=labels.index)
				labels  = pd.DataFrame( dataset.labels , columns=columns)

			self.dataset = dataset

		# Loading the data using torchxrayvision package
		self.dataset = self.load_raw_database()

		# Relabeling it with respect to the model pathologies
		self.relabel_raw_database()

		# Updating the empty parent labels with the child labels
		if self.config.datasetName in [DatasetNames.PC , DatasetNames.NIH]:
			self.update_empty_parent_class_based_on_its_children_classes()

		# Selecting non-null samples for impacted pathologies
		if self.config.NotNull_Samples:
			selecting_non_null_samples()

		#separate train & test
		dataset_train, dataset_test = train_test_split()

		# Creating the data_loader
		data_loader_args = {"batch_size" : self.config.batch_size,
							"shuffle"    : self.config.shuffle,
							"num_workers": self.config.num_workers,
							"pin_memory" : USE_CUDA		}
		data_loader_train = torch.utils.data.DataLoader(dataset_train, **data_loader_args)
		data_loader_test  = torch.utils.data.DataLoader(dataset_test , **data_loader_args )

		self.train = Data(dataset=dataset_train, data_loader=data_loader_train)
		self.test  = Data(dataset=dataset_test , data_loader=data_loader_test )


	@staticmethod
	def get_dataset_pathologies(datasetName):
		pathologies_dict = dict(
				NIH      = ["Atelectasis"                , "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"                                                                                                                                                                                                                                                                                   ],
				RSNA     = ["Pneumonia"                  , "Lung Opacity"                                                                                                                                                                                                                                                                                                                                                                                                                                                 ],
				PC       = ["Atelectasis"                , "Consolidation" , "Infiltration" , "Pneumothorax" , "Edema" , "Emphysema" , "Fibrosis" , "Effusion" , "Pneumonia" , "Pleural_Thickening" , "Cardiomegaly" , "Nodule" , "Mass" , "Hernia", "Fracture", "Granuloma", "Flattened Diaphragm", "Bronchiectasis", "Aortic Elongation", "Scoliosis", "Hilar Enlargement", "Tuberculosis", "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis", "Hemidiaphragm Elevation", "Support Devices", "Tube'"], # the Tube' is intentional
				CheX     = ["Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion" , "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"                                                                                                                                                                                                                                                             ],
				MIMIC    = ["Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion" , "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"                                                                                                                                                                                                                                                             ],
				Openi    = ["Atelectasis"                , "Fibrosis" , "Pneumonia" , "Effusion" , "Lesion" , "Cardiomegaly" , "Calcified Granuloma" , "Fracture" , "Edema" , "Granuloma" , "Emphysema" , "Hernia" , "Mass" , "Nodule", "Opacity", "Infiltration", "Pleural_Thickening", "Pneumothorax"                                                                                                                                                                                                                   ],
				NLMTB    = ["Tuberculosis"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ],
				VinBrain = ['Aortic enlargement'         , 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Lesion', 'Effusion', 'Pleural_Thickening', 'Pneumothorax', 'Pulmonary Fibrosis'                                                                                                                                                                                                                                                        ]
				)
		return pathologies_dict[datasetName]

	@property
	def xrv_default_pathologies(self):
		return  xrv.datasets.default_pathologies

	def _path_csv_files(self) -> pathlib.Path:
		csv_path_dict = {
				DatasetNames.NIH     : 'NIH/Data_Entry_2017.csv',
				DatasetNames.RSNA    : None,
				DatasetNames.PC      : 'PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',
				DatasetNames.CheX    : f'CheX/CheXpert-v1.0-small/{self.config.dataset_data_mode}.csv',
				DatasetNames.MIMIC   : 'MIMIC/mimic-cxr-2.0.0-chexpert.csv.gz',
				DatasetNames.Openi   : 'Openi/nlmcxr_dicom_metadata.csv',
				DatasetNames.NLMTB   : None,
				DatasetNames.VinBrain: f'VinBrain/dicom/{self.config.dataset_data_mode}.csv',
		}

		return self.config.PATH_DATASETS / csv_path_dict.get(self.config.datasetName)

	def _path_meta_data_csv_files(self) -> pathlib.Path | None:

		meta_csv_path_dict = dict(MIMIC='MIMIC/mimic-cxr-2.0.0-metadata.csv.gz') # I don't have this csv file
		return self.config.PATH_DATASETS / meta_csv_path_dict.get(self.config.datasetName)

	def _path_dataset(self) -> pathlib.Path:
		dataset_dir_dict = {
				DatasetNames.NIH      : 'NIH/images-224',
				DatasetNames.RSNA     : None,
				DatasetNames.PC       : 'PC/images-224',
				DatasetNames.CheX     : 'CheX/CheXpert-v1.0-small',
				DatasetNames.MIMIC    : 'MIMIC/re_512_3ch',
				DatasetNames.Openi    : 'Openi/NLMCXR_png',
				DatasetNames.NLMTB    : None,
				DatasetNames.VinBrain : f'VinBrain/{self.config.dataset_data_mode}'
				}

		return self.config.PATH_DATASETS / dataset_dir_dict.get(self.config.datasetName)

	@staticmethod
	def format_image(img):
		transform = torchvision.transforms.Compose( [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size=224, engine="cv2")])
		img = transform(img)
		img = torch.from_numpy(img)
		return img

	@property
	def available_datasets(self):
		return [DatasetNames.CheXPERT, DatasetNames.NIH, DatasetNames.PC]

	@property
	def labels(self):
		return pd.DataFrame(self.dataset.labels, columns=self.dataset.pathologies)





