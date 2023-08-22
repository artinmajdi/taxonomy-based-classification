import argparse
import itertools
import pathlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader as torch_DataLoader

import torchxrayvision as xrv
from taxonomy.utilities.params import DataModes, DatasetNames, ExperimentStageNames, HyperparameterNames, \
	TechniqueNames, ThreshTechList

USE_CUDA = torch.cuda.is_available()


@dataclass
class Nodes:
	CLASSES          : set[str]
	NON_NULL         : set[str] 		   = field(default_factory = set)
	IMPACTED         : set[str] 		   = field(default_factory = set)
	TAXONOMY         : dict[str, set[str]] = field(default_factory = dict)
	EXIST_IN_TAXONOMY: set[str] 	       = field(default_factory = set)

	def __post_init__(self):
		self.TAXONOMY          = self._taxonomy()
		self.EXIST_IN_TAXONOMY = self._exist_in_taxonomy()

	@staticmethod
	def _default_taxonomy() -> dict[str, set[str]]:
		return {'Lung Opacity': {'Pneumonia', 'Atelectasis', 'Consolidation', 'Lung Lesion', 'Edema', 'Infiltration'}, 'Enlarged Cardiomediastinum': {'Cardiomegaly'}}

	def _taxonomy(self):  # sourcery skip: dict-comprehension
		UT = {}
		for parent in self._default_taxonomy():
			if parent in self.CLASSES:
				UT[parent] = self._default_taxonomy()[parent].intersection(self.CLASSES)

		return UT or None

	def _exist_in_taxonomy(self) -> set[str]:

		# Adding the parent classes
		eit = set(self.TAXONOMY.keys())

		# Adding the children classes
		for value in self.TAXONOMY.values():
			eit.update(value)

		return eit

	@property
	def node_thresh_tuple(self):
		return list(itertools.product(self.IMPACTED, ThreshTechList))

	def get_children_of(self, parent: str) -> set[str]:
		return self.TAXONOMY.get(parent, set())

	def get_parent_of(self, child: str) -> str | None:
		return next((parent for parent in self.TAXONOMY if child in self.TAXONOMY[parent]), None)


@dataclass
class NodeData:
	CLASSES: set[str]

	def __post_init__(self):
		self.graph = self._construct_graph()

	@cached_property
	def nodes(self) -> Nodes:
		return Nodes(CLASSES=self.CLASSES)

	def _construct_graph(self) -> nx.DiGraph:
		graph = nx.DiGraph(self.nodes.TAXONOMY)
		graph.add_nodes_from(self.nodes.CLASSES)
		return graph

	def add_hyperparameters_to_node(self, parent_node: str, child_node: str, hyperparameter: dict[HyperparameterNames, float]):
		for hp_name in hyperparameter:
			self.graph.edges[parent_node, child_node][hp_name] = hyperparameter[hp_name]

	def get_hyperparameters_of_node(self, parent_node: str, child_node: str) -> dict[HyperparameterNames, float]:
		return self.graph.edges[parent_node, child_node]

	def add_hierarchy_penalty(self, node: str, penalty: float):
		self.graph.nodes[node]['hierarchy_penalty'] = penalty

	def get_hierarchy_penalty(self, node: str) -> float:
		return self.graph.nodes[node].get('hierarchy_penalty', None)

	def show(self, package='networkx'):

		import plotly.graph_objs as go
		import seaborn as sns

		def plot_networkx_digraph(graph):
			pos = nx.spring_layout(graph)
			edge_x = []
			edge_y = []
			for edge in graph.edges():
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

			node_x = [pos[node][0] for node in graph.nodes()]
			node_y = [pos[node][1] for node in graph.nodes()]

			colorbar = dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')
			marker = dict(showscale=True, colorscale='Viridis', reversescale=True, color=[], size=10, line_width=2,
						  colorbar=colorbar)

			node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=marker)

			node_adjacency = []
			node_text = []
			for node, adjacency in enumerate(graph.adjacency()):
				node_adjacency.append(len(adjacency[1]))
				node_text.append(f'{adjacency[0]} - # of connections: {len(adjacency[1])}')

			node_trace.marker.color = node_adjacency
			node_trace.text = node_text

			# Add node labels
			annotations = []
			for node in graph.nodes():
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
			nx.draw(self.graph, with_labels=True)

		elif package == 'plotly':
			plot_networkx_digraph(self.graph)


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


@dataclass
class Data:
	dataset    : xrv.datasets.Dataset
	data_loader: torch_DataLoader
	labels     : Labels    = field(default_factory = lambda: None)
	dataMode   : DataModes = field(default_factory = lambda: None)

	def __post_init__(self):
		self.labels = Labels(LABEL_SET=self.dataset.labels, CLASSES=self.dataset.pathologies)


@dataclass
class Metrics:
	ACC: pd.DataFrame = field(default_factory = lambda: None)
	AUC: pd.DataFrame = field(default_factory = lambda: None)
	F1 : pd.DataFrame = field(default_factory = lambda: None)
	THRESHOLD: dict[ThreshTechList, pd.DataFrame] = field(default_factory = lambda: None)


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


@dataclass
class LoadChestXrayDatasets:
	config  : argparse.Namespace   = field(default_factory = lambda: None)
	dataset : xrv.datasets.Dataset = field(default_factory = lambda: None)
	train   : Data                 = field(default_factory = lambda: None)
	test    : Data                 = field(default_factory = lambda: None)
	nodeData: NodeData             = field(default_factory = lambda: None)

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

		def _path_csv_files() -> pathlib.Path:
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

		def _path_meta_data_csv_files() -> pathlib.Path | None:
			meta_csv_path_dict = dict(MIMIC='MIMIC/mimic-cxr-2.0.0-metadata.csv.gz') # I don't have this csv file
			return self.config.PATH_DATASETS / meta_csv_path_dict.get(self.config.datasetName)

		def _path_dataset() -> pathlib.Path:
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

		imgpath   = _path_dataset()
		views     = self.config.views
		csvpath   = _path_csv_files()
		transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size = 224, engine = "cv2")])

		params_config = {
				DatasetNames.NIH       : dict(imgpath=imgpath, views=views),
				DatasetNames.PC        : dict(imgpath=imgpath, views=views),
				DatasetNames.CHEXPERT  : dict(imgpath=imgpath, views=views, transform=transform, csvpath=csvpath),
				DatasetNames.MIMIC     : dict(imgpath=imgpath, views=views, transform=transform, csvpath=csvpath, metacsvpath=_path_meta_data_csv_files()),
				DatasetNames.Openi     : dict(imgpath=imgpath, views=views, transform=transform),
				DatasetNames.VinBrain  : dict(imgpath=imgpath, views=views, csvpath=csvpath),
				DatasetNames.RSNA      : dict(imgpath=imgpath, views=views, transform=transform),
				DatasetNames.NIH_Google: dict(imgpath=imgpath, views=views)
				}

		dataset_config = {
				DatasetNames.NIH       : xrv.datasets.NIH_Dataset,
				DatasetNames.PC        : xrv.datasets.PC_Dataset,
				DatasetNames.CHEXPERT  : xrv.datasets.CheX_Dataset,
				DatasetNames.MIMIC     : xrv.datasets.MIMIC_Dataset,
				DatasetNames.Openi     : xrv.datasets.Openi_Dataset,
				DatasetNames.VinBrain  : xrv.datasets.VinBrain_Dataset,
				DatasetNames.RSNA      : xrv.datasets.RSNA_Pneumonia_Dataset,
				DatasetNames.NIH_Google: xrv.datasets.NIH_Google_Dataset
				}

		self.dataset = dataset_config.get(self.config.datasetName)(**params_config.get(self.config.datasetName))

	def relabel_raw_database(self):
		from taxonomy.utilities.model import LoadModelXRV

		# Adding the PatientID if it doesn't exist
		# if "patientid" not in self.dataset.csv:
		# 	self.dataset.csv["patientid"] = [ f"{self.dataset.__class__.__name__}-{i}" for i in range(len(self.dataset)) ]

		# Filtering the dataset
		# self.dataset.csv = self.dataset.csv[self.dataset.csv['Frontal/Lateral'] == 'Frontal'].reset_index(drop=False)

		# Aligning labels to have the same order as the pathologies' argument.
		xrv.datasets.relabel_dataset( pathologies=LoadModelXRV.model_classes(self.config), dataset=self.dataset, silent=self.config.silent)

	def update_empty_parent_class_based_on_its_children_classes(self):

		labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

		for parent, children in self.nodeData.nodes.TAXONOMY.items():

			# Checking if the parent class existed in the original pathologies in the dataset. Will only replace its values if all its labels are NaN
			if labels[parent].value_counts().values.shape[0] == 0:

				print(f"Parent class: {parent} is not in the dataset. replacing its true values according to its children presence.")

				# Initializing the parent label to 0
				labels[parent] = 0

				# If at-least one of the children has a label of 1, then the parent label is 1
				labels[parent][ labels[children].sum(axis=1) > 0 ] = 1

		self.dataset.labels = labels.values

	def train_test_split(self):
		labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

		idx_train = labels.sample(frac=self.config.train_test_ratio).index
		d_train = xrv.datasets.SubsetDataset(self.dataset, idxs=idx_train)

		idx_test = labels.drop(idx_train).index
		d_test = xrv.datasets.SubsetDataset(self.dataset, idxs=idx_test)

		return d_train, d_test

	def _selecting_non_null_samples(self):
		"""  Selecting non-null samples for impacted pathologies  """

		labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

		for parent in self.nodeData.nodes.TAXONOMY:

			# Extracting the samples with a non-null value for the parent truth label
			labels  = labels[ ~labels[parent].isna() ]
			self.dataset = xrv.datasets.SubsetDataset(self.dataset, idxs=labels.index)
			labels  = pd.DataFrame( self.dataset.labels , columns=self.dataset.pathologies)

			# Extracting the samples, where for each parent, at least one of their children has a non-null truth label
			labels  = labels[ (~labels[ self.nodeData.nodes.TAXONOMY[parent] ].isna()).sum(axis=1) > 0 ]
			self.dataset = xrv.datasets.SubsetDataset(self.dataset, idxs=labels.index)

	def load(self):

		# Loading the data using torchxrayvision package
		self.load_raw_database()

		# Relabeling it with respect to the model pathologies
		self.relabel_raw_database()

		# Updating the empty parent labels with the child labels
		if self.config.datasetName in [DatasetNames.PC , DatasetNames.NIH]:
			self.update_empty_parent_class_based_on_its_children_classes()

		# Selecting non-null samples for impacted pathologies
		self._selecting_non_null_samples()

		#separate train & test
		dataset_train, dataset_test = self.train_test_split()

		# Creating the data_loader
		data_loader_args = {key: getattr(self.config, key) for key in ['batch_size', 'shuffle', 'num_workers']}
		data_loader_train = torch.utils.data.DataLoader(dataset_train, pin_memory=USE_CUDA , **data_loader_args)
		data_loader_test  = torch.utils.data.DataLoader(dataset_test , pin_memory=USE_CUDA , **data_loader_args )

		self.train = Data(dataset=dataset_train, data_loader=data_loader_train)
		self.test  = Data(dataset=dataset_test , data_loader=data_loader_test )

	@property
	def xrv_default_pathologies(self):
		return  xrv.datasets.default_pathologies

	@staticmethod
	def format_image(img):
		transform = torchvision.transforms.Compose( [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size=224, engine="cv2")])
		img = transform(img)
		img = torch.from_numpy(img)
		return img

	@property
	def available_datasets(self):
		return [DatasetNames.CHEXPERT, DatasetNames.NIH, DatasetNames.PC]
