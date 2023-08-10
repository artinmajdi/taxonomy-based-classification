import argparse
import concurrent.futures
import contextlib
import itertools
import json
import multiprocessing
import os
import pathlib
import pickle
import resource
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objs as go
import seaborn as sns
import sklearn
import torch
import torchvision
import umap
from hyperopt import fmin, hp, tpe
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

import torchxrayvision as xrv
from taxonomy.data_classes import Data, Findings, Labels
from taxonomy.fixed_params import ExperimentSTAGE, DatasetList, ThreshTechList, DataModes, MethodNames

# region
USE_CUDA = torch.cuda.is_available()

class Hierarchy:

	taxonomy_structure = { 'Lung Opacity'              : ['Pneumonia', 'Atelectasis', 'Consolidation', 'Lung Lesion', 'Edema', 'Infiltration'],
							'Enlarged Cardiomediastinum': ['Cardiomegaly']}
	# 'Infiltration'              : ['Consolidation'],
	# 'Consolidation'             : ['Pneumonia'],

	class OUTPUT(NamedTuple):
		hierarchy_penalty: pd.DataFrame
		metrics: Union[dict, pd.DataFrame]
		data: Union[ pd.DataFrame, Dict[str, pd.DataFrame] ]

	def __init__(self, classes=None):

		if classes is None:
			classes = []

		self.classes = list( set(classes) )

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
			if classes and len( classes ) > 0: self.G.add_nodes_from( classes )
			if hierarchy and len( hierarchy ) > 0: self.G.add_edges_from( hierarchy )

		def add_hyperparameters_to_graph():
			for parent_node, child_node in self.G.edges:
				self.G.edges[parent_node, child_node]['hyperparameters'] = { x: hyperparameters[x][child_node].copy() for x in ThreshTechList }

		def graph_add_findings_original_to_nodes(findings: dict):

			# Loop over all classes (aka nodes)
			for n in findings['truth'].columns:

				data = pd.DataFrame(
					dict( truth=findings['truth'][n], pred=findings['pred'][n], logit=findings['logit'][n],
						  loss=findings['loss'][n] ) )

				metrics = { }
				for x in ThreshTechList:
					metrics[x] = findings['metrics'][x, n]

				# Adding the findings to the graph node
				self.G.nodes[n][ExperimentSTAGE.ORIGINAL] = dict( data=data, metrics=metrics )

		def graph_add_findings_new_to_nodes(findings: dict):

			# Loop over all classes (aka nodes)
			for n in findings['truth'].columns:

				# Merging the pred, truth, and loss into a dataframe
				data, metrics, hierarchy_penalty = { }, { }, pd.DataFrame()
				for x in ThreshTechList:
					data[x] = pd.DataFrame(
						dict( truth=findings['truth'][n], pred=findings['pred'][x, n], loss=findings['loss'][x, n] ) )
					metrics[x] = findings['metrics'][x, n]
					hierarchy_penalty[x] = findings['hierarchy_penalty'][x, n].values

				# Adding the findings to the graph node
				self.G.nodes[n][ExperimentSTAGE.NEW] = dict( data=data, metrics=metrics )

				# Updating the graph with the hierarchy_penalty for the current node
				self.G.nodes[n]['hierarchy_penalty'] = hierarchy_penalty


		if classes           : add_nodes_and_edges()
		if hyperparameters   : add_hyperparameters_to_graph()
		if findings_new      : graph_add_findings_new_to_nodes(findings_new)
		if findings_original : graph_add_findings_original_to_nodes(findings_original)

	def add_hyperparameters_to_node(self, parent, child, a=1.0, b=0.0):  # type: (str, str, float, float) -> None
		self.G.edges[parent, child]['a'] = a
		self.G.edges[parent, child]['b'] = b

	def graph_get_taxonomy_hyperparameters(self, parent_node, child_node):  # type: (str, str) -> Tuple[float, float]
		return self.G.edges[parent_node, child_node]['hyperparameters']

	@ staticmethod
	def get_node_original_results_for_child_and_parent_nodes(G, node, thresh_technique='ROC'): # type: (nx.DiGraph, str, str) -> Tuple[Dict[str, Any], Dict[str, Any]]

		# The predicted probability p ,  true label y, optimum threshold th, and loss for node class
		child_data =Hierarchy.get_findings_for_node(G=G , node=node, thresh_technique=thresh_technique, WHICH_RESULTS=ExperimentSTAGE.ORIGINAL)

		# The predicted probability p ,  true label y, optimum threshold th, and loss for parent class
		parent_node = Hierarchy.get_parent_node(G=G , node=node)
		parent_data = Hierarchy.get_findings_for_node( G=G , node=parent_node , thresh_technique=thresh_technique, WHICH_RESULTS=ExperimentSTAGE.ORIGINAL) if parent_node else None

		return child_data, parent_data

	@staticmethod
	def get_findings_for_node(G, node, thresh_technique, WHICH_RESULTS): # type: (nx.DiGraph, str, str, ExperimentSTAGE) -> Hierarchy.OUTPUT

		WR = WHICH_RESULTS
		TT = thresh_technique

		node_data = G.nodes[node]

		if WR == ExperimentSTAGE.ORIGINAL:
			data    = node_data[WR]['data'   ]
			metrics = node_data[WR]['metrics'][TT]
			hierarchy_penalty  = None
		elif WR == ExperimentSTAGE.NEW:
			data    = node_data[WR]['data'   ][TT]
			metrics = node_data[WR]['metrics'][TT]
			hierarchy_penalty  = node_data['hierarchy_penalty']
		else:
			raise ValueError('WHICH_RESULTS must be either ORIGINAL or NEW')

		return Hierarchy.OUTPUT( data=data, metrics=metrics, hierarchy_penalty=hierarchy_penalty )

	@staticmethod
	def get_parent_node(G, node): # type: (nx.DiGraph, str) -> str
		parent_node = nx.dag.ancestors(G, node)
		return list(parent_node)[0] if len(parent_node)>0 else None

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

		return LPI.intersection( set(self.classes) )

	def create_hierarchy(self, classes: list):

		UT = defaultdict(list)

		# Looping through all parent classes in the DEFAULT taxonomy
		for parent in Hierarchy.taxonomy_structure.keys():

			# Check if the parent class of our DEFAULT taxonomy exist in the dataset
			if parent in classes:
				for child in Hierarchy.taxonomy_structure[parent]:

					# Checks if the child classes of found parent class existi in the dataset
					if child in classes:
						UT[parent].append(child)

		return UT or None

	@property
	def parent_child_df(self):
		df = pd.DataFrame(columns=['parent' , 'child'])
		for p in self.hierarchy.keys():
			for c in self.hierarchy[p]:
				df = df.append({'parent': p, 'child': c}, ignore_index=True)
		return df

	def show(self, package='networkx', **kwargs):

		def plot_networkx_digraph(G):
			pos = nx.spring_layout( G )
			edge_x = []
			edge_y = []
			for edge in G.edges():
				x0, y0 = pos[edge[0]]
				x1, y1 = pos[edge[1]]
				edge_x.append( x0 )
				edge_x.append( x1 )
				edge_x.append( None )
				edge_y.append( y0 )
				edge_y.append( y1 )
				edge_y.append( None )

			edge_trace = go.Scatter( x=edge_x, y=edge_y, line=dict( width=0.5, color='#888' ), hoverinfo='none', mode='lines' )

			node_x = [pos[node][0] for node in G.nodes()]
			node_y = [pos[node][1] for node in G.nodes()]

			node_trace = go.Scatter( x=node_x, y=node_y, mode='markers', hoverinfo='text',
									 marker=dict(showscale=True, colorscale='Viridis', reversescale=True, color=[], size=10, line_width=2, colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')) )

			node_adjacencies = []
			node_text        = []
			for node, adjacencies in enumerate( G.adjacency() ):
				node_adjacencies.append( len( adjacencies[1] ) )
				node_text.append( f'{adjacencies[0]} - # of connections: {len( adjacencies[1] )}' )

			node_trace.marker.color = node_adjacencies
			node_trace.text         = node_text

			# Add node labels
			annotations = []
			for node in G.nodes():
				x, y = pos[node]
				annotations.append(go.layout.Annotation(text=str( node ), x=x, y=y, showarrow=False, font=dict( size=10, color='black' )))

			fig = go.Figure( data=[edge_trace, node_trace],
							 layout=go.Layout( title='Networkx DiGraph', showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), annotations=annotations ))
			fig.show()

		if package == 'networkx':
			sns.set_style("whitegrid")
			nx.draw(self.G, with_labels=True)

		elif package == 'plotly':
			plot_networkx_digraph(self.G)

class SaveFile:
	def __init__(self, path):
		self.path = path

	def load(self, index_col=None, header=None):

		if self.path.exists():

			if self.path.suffix == '.pkl':
				with open(self.path, 'rb') as f:
					return pickle.load(f)

			elif self.path.suffix == '.csv':
				return pd.read_csv(self.path)

			elif self.path.suffix == '.xlsx':
				return pd.read_excel(self.path, index_col=index_col, header=header)

		return None

	def dump(self, file, index=False):

		self.path.parent.mkdir(parents=True, exist_ok=True)

		if self.path.suffix == '.pkl':
			with open(self.path, 'wb') as f:
				pickle.dump(file, f)

		elif self.path.suffix == '.csv':
			file.to_csv(self.path, index=index)

		elif self.path.suffix == '.xlsx':
			file.to_excel(self.path, index=index)

class SaveFigure:

	def __init__(self, config, path='', SAVE_ALL_FORMATS=True):
		self.config   = config
		self.path    = self.config.local_path.joinpath(path)

	def load(self):
		return plt.imread(self.path) if self.path.exists() else None

	def dump(self):
		self.path.parent.mkdir(parents=True, exist_ok=True)

		for format in ['png', 'eps', 'svg', 'pdf']:

			path = self.path.with_suffix(f'.{format}')
			plt.savefig(path, format=format, dpi=300)

class LoadSaveFindings:
	def __init__(self, config, relative_path: Union[str, pathlib.Path]='sth.pkl'):
		""" relative_path can be one of [findings_original , findings_new, hyperparameters.pkl]"""

		self.config   = config
		self.relative_path = relative_path
		self.path = self.config.local_path.joinpath(relative_path)

	def save(self, data, **kwargs):
		SaveFile(self.path).dump(data, **kwargs)

	def load(self, source='load_MLFlow', run_name=None, run_id=None, **kwargs):
		return SaveFile(self.path).load(**kwargs)

class LoadChestXrayDatasets:

	def __init__(self, config: argparse.Namespace, pathologies_in_model: List[str]=None) -> None:

		self.d_data: Optional[xrv.datasets.CheX_Dataset]  = None
		self.pathologies_in_model = pathologies_in_model or []

		self.train = Data(DataModes.TRAIN)
		self.test  = Data(DataModes.TEST)

		self.config = config
		self.config.dataset_name = self.fix_dataset_name(config.dataset_name)

	def load_raw_database(self):
		"""
			# RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
				Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
				Annotations of Possible Pneumonia.	Shih, George, Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
				More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
				Challenge site:	https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
				JPG files stored here: 	https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
				
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
				A relabelling of a subset of images from the NIH dataset.  The data tables should
				be applied against an NIH download.  A test and validation split are provided in the
				original.  They are combined here, but one or the other can be used by providing
				the original csv to the csvpath argument.

				Chest Radiograph Interpretation with Deep Learning Models: Assessment with
				Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
				Anna Majkowska,. Radiology 2020		https://pubs.rsna.org/doi/10.1148/radiol.2019191293
				NIH data can be downloaded here:	https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
		"""

		assert self.dataset_path.exists(), "Dataset directory does not exist!"
		imgpath   = str(self.dataset_path)
		views     = self.config.views
		csvpath   = str(self.csv_path)
		transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size = 224, engine = "cv2")])

		params_config = {
			'NIH'       : dict(imgpath = imgpath, views = views),
			'PC'        : dict(imgpath = imgpath, views = views),
			'CheX'      : dict(imgpath = imgpath, views = views , transform = transform , csvpath = csvpath),
			'MIMIC'     : dict(imgpath = imgpath, views = views , transform = transform , csvpath = csvpath , metacsvpath = str(self.meta_csv_path)),
			'Openi'     : dict(imgpath = imgpath, views = views , transform = transform),
			'VinBrain'  : dict(imgpath = imgpath, views = views , csvpath   = csvpath),
			'RSNA'      : dict(imgpath = imgpath, views = views , transform = transform),
			'NIH_Google': dict(imgpath = imgpath, views = views)
		}
		
		dataset_config = {
			'NIH'       : xrv.datasets.NIH_Dataset,
			'PC'        : xrv.datasets.PC_Dataset,
			'CheX'      : xrv.datasets.CheX_Dataset,
			'MIMIC'     : xrv.datasets.MIMIC_Dataset,
			'Openi'     : xrv.datasets.Openi_Dataset,
			'VinBrain'  : xrv.datasets.VinBrain_Dataset,
			'RSNA'      : xrv.datasets.RSNA_Pneumonia_Dataset,
			'NIH_Google': xrv.datasets.NIH_Google_Dataset
		}

		d_name = self.config.dataset_name
		if d_name in dataset_config:
			self.d_data = dataset_config.get(d_name)(**params_config.get(d_name))


	def relabel_raw_database(self):
		# Adding the PatientID if it doesn't exist
		if "patientid" not in self.d_data.csv:
			self.d_data.csv["patientid"] = [ f"{self.d_data.__class__.__name__}-{i}" for i in range(len(self.d_data)) ]

		# Filtering the dataset
		# self.d_data.csv = self.d_data.csv[self.d_data.csv['Frontal/Lateral'] == 'Frontal'].reset_index(drop=False)

		# Aligning labels to have the same order as the pathologies' argument.
		if len(self.pathologies_in_model) > 0:
			xrv.datasets.relabel_dataset( pathologies=self.pathologies_in_model, dataset=self.d_data, silent=self.config.silent)

	def update_empty_parent_class_based_on_its_children_classes(self):

		columns = self.d_data.pathologies
		labels  = pd.DataFrame( self.d_data.labels , columns=columns)

		# This will import the child_dict
		child_dict = Hierarchy(classes=self.d_data.pathologies).child_dict

		for parent, children in child_dict.items():
			
			# Checking if the parent class existed in the original pathologies in the dataset. will only replace its values if all its labels are NaN
			if labels[parent].value_counts().values.shape[0] == 0:
				if not self.config.silent: print(f"Parent class: {parent} is not in the dataset. replacing its true values according to its children presence.")

				# Initializing the parent label to 0
				labels[parent] = 0

				# If at-least one of the children has a label of 1, then the parent label is 1
				labels[parent][ labels[children].sum(axis=1) > 0 ] = 1

		self.d_data.labels = labels.values

	def load(self):

		def train_test_split():
			labels  = pd.DataFrame( self.d_data.labels , columns=self.d_data.pathologies)

			idx_train = labels.sample(frac=self.config.train_test_ratio).index
			d_train = xrv.datasets.SubsetDataset(self.d_data, idxs=idx_train)

			idx_test = labels.drop(idx_train).index
			d_test = xrv.datasets.SubsetDataset(self.d_data, idxs=idx_test)

			return d_train, d_test

		def selecting_not_null_samples():
			"""  Selecting non-null samples for impacted pathologies  """

			dataset = self.d_data
			columns = self.d_data.pathologies
			labels  = pd.DataFrame( dataset.labels , columns=columns)

			# This will import the child_dict
			child_dict = Hierarchy(classes=self.d_data.pathologies).child_dict

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

			self.d_data = dataset

		# Loading the data using torchxrayvision package
		self.load_raw_database()

		# Relabeling it with respect to the model pathologies
		self.relabel_raw_database()

		# Updating the empty parent labels with the child labels
		if self.config.dataset_name in [ 'PC' , 'NIH']:
			self.update_empty_parent_class_based_on_its_children_classes()

		# Selecting non-null samples for impacted pathologies
		if self.config.NotNull_Samples:  selecting_not_null_samples()

		#separate train & test
		self.train.d_data, self.test.d_data = train_test_split()

		# Creating the data_loader
		data_loader_args = {"batch_size" : self.config.batch_size,
							"shuffle"    : self.config.shuffle,
							"num_workers": self.config.num_workers,
							"pin_memory" : USE_CUDA		}
		self.train.data_loader = torch.utils.data.DataLoader(self.train.d_data, **data_loader_args)
		self.test.data_loader  = torch.utils.data.DataLoader(self.test.d_data , **data_loader_args )

	@staticmethod
	def fix_dataset_name(dataset_name):

		# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031
		if dataset_name.lower() in ('chex', 'chexpert'): return 'CheX'

		# National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
		elif dataset_name.lower() in ('nih',): return 'NIH'

		elif dataset_name.lower() in ('openi',): return 'Openi'

		# PadChest: A large chest thresh_technique-ray image dataset with multi-label annotated reports. https://arxiv.org/abs/1901.07441
		elif dataset_name.lower() in ('pc', 'padchest'): return 'PC'

		# VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
		elif dataset_name.lower() in ('vinbrain',): return 'VinBrain'

		# RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
		elif dataset_name.lower() in ('rsna',): return 'RSNA'

		# MIMIC-CXR (MIT)
		elif dataset_name.lower() in ('mimic',): return 'MIMIC'

		# National Library of Medicine Tuberculosis Datasets. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/
		elif dataset_name.lower() in ('nlmtb',): return 'NLMTB'

		# SIIM Pneumothorax Dataset. https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
		elif dataset_name.lower() in ('siim',): return 'SIIM'

		# A relabelling of a subset of NIH images from: https://pubs.rsna.org/doi/10.1148/radiol.2019191293
		elif  dataset_name.lower() in ('nih_google',): return 'NIH_Google'

	@staticmethod
	def get_dataset_pathologies(dataset_name):
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
		return pathologies_dict[dataset_name]

	@property
	def xrv_default_pathologies(self):
		return  xrv.datasets.default_pathologies # [ 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum' ]

	@property
	def csv_path(self) -> pathlib.Path:
		csv_path_dict = dict(   NIH      = 'NIH/Data_Entry_2017.csv',
								RSNA     = None,
								PC       = 'PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',
								CheX     = f'CheX/CheXpert-v1.0-small/{self.config.dataset_data_mode}.csv',
								MIMIC    = 'MIMIC/mimic-cxr-2.0.0-chexpert.csv.gz',
								Openi    = 'Openi/nlmcxr_dicom_metadata.csv',
								NLMTB    = None,
								VinBrain = f'VinBrain/dicom/{self.config.dataset_data_mode}.csv',
								)

		return self.config.dataset_path.joinpath(csv_path_dict[self.config.dataset_name])

	@property
	def meta_csv_path(self) -> pathlib.Path:
		meta_csv_path_dict = dict(MIMIC='MIMIC/mimic-cxr-2.0.0-metadata.csv.gz') # I don't have this csv file
		if self.config.dataset_name in meta_csv_path_dict:
			return self.config.dataset_path.joinpath(meta_csv_path_dict.get(self.config.dataset_name))
		return None # type: ignore

	@property
	def dataset_path(self) -> pathlib.Path:
		dataset_dir_dict = dict(
								NIH     ='NIH/images-224',
								RSNA    = None,
								PC      = 'PC/images-224',
								CheX    = 'CheX/CheXpert-v1.0-small',
								MIMIC   = 'MIMIC/re_512_3ch',
								Openi   = 'Openi/NLMCXR_png',
								NLMTB   = None,
								VinBrain= f'VinBrain/{self.config.dataset_data_mode}' )

		return self.config.dataset_path.joinpath(dataset_dir_dict[self.config.dataset_name])

	@staticmethod
	def format_image(img):
		transform = torchvision.transforms.Compose( [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size=224, engine="cv2")])
		img = transform(img)
		img = torch.from_numpy(img)
		return img

	@property
	def all_datasets_available(self):
		return ['CheX' ,'PC', 'Openi',  'NIH']

	@property
	def labels(self):
		return pd.DataFrame(self.d_data.labels, columns=self.d_data.pathologies)


	@classmethod
	def get_dataset_unfiltered(cls, update_empty_parent_class=False , **kwargs):
		config = reading_user_input_arguments(**kwargs)
		model = LoadModelXRV(config).load()
		LD = cls(config=config, pathologies_in_model=model.pathologies)
		LD.load_raw_database()
		LD.relabel_raw_database()

		if update_empty_parent_class:
			LD.update_empty_parent_class_based_on_its_children_classes()

		return LD


class LoadModelXRV:
	def __init__(self, config):
		self.model 		: Optional[torch.nn.Module] = None
		self.model_name : str 					    = self.fix_model_name(config.model_name)
		self.config 	 : argparse.Namespace	     = config

	def load(self, op_threshs: bool=False) -> torch.nn.Module:
		""" 224x224 models """
		model_name = self.model_name

		
		get_model = lambda pre_trained_model_weights: xrv.models.DenseNet( weights=pre_trained_model_weights, apply_sigmoid=False )

		if   model_name == 'NIH'      : model = get_model('densenet121-res224-nih')
		elif model_name == 'RSNA'     : model = get_model('densenet121-res224-rsna')
		elif model_name == 'PC'       : model = get_model('densenet121-res224-pc')
		elif model_name == 'CheX'     : model = get_model('densenet121-res224-chex')
		elif model_name == 'MIMIC_NB' : model = get_model('densenet121-res224-mimic_nb')
		elif model_name == 'MIMIC_CH' : model = get_model('densenet121-res224-mimic_ch')
		elif model_name == 'ALL_224'  : model = get_model('densenet121-res224-all')
		elif model_name == 'ALL_512'  : model = get_model('resnet50-res512-all')
		elif model_name == 'baseline_jfhealthcare' : model = xrv.baseline_models.jfhealthcare.DenseNet()
		elif model_name == 'baseline_CheX'         : model = xrv.baseline_models.chexpert.DenseNet(weights_zip=self.config.path_baseline_CheX_weights.as_posix())
		elif model_name == 'Openi'                 : model = None
		elif model_name == 'NLMTB'                 : model = None
		elif model_name == 'VinBrain'              : model = None
		else									   : model = None

		if not op_threshs:
			model.op_threshs = None

		# Enabling CUDA
		if USE_CUDA:
			model.cuda()

		self.model = model

		return self.model

	@staticmethod
	def fix_model_name(model_name) -> str:
		mapping = dict(nih 					 ='NIH',
						rsna                  = 'RSNA',
						pc                    = 'PC'                   , padchest = 'PC',
						chex                  = 'CheX'                 , chexpert = 'CheX',
						mimic_nb              = 'MIMIC_NB',
						mimic_ch              = 'MIMIC_CH',
						all_224               = 'ALL_224',
						all_512               = 'ALL_512',
						baseline_jfhealthcare = 'baseline_jfhealthcare',
						baseline_chex         = 'baseline_CheX',
						openi                 = 'Openi',
						nlmtb                 = 'NLMTB',
						vinbrain              = 'VinBrain')
		return mapping[model_name.lower()]

	@property
	def xrv_op_threshs(self):
		return xrv.models.model_urls[self.model_name.lower()]['op_threshs']

	@property
	def xrv_labels(self):
		return xrv.models.model_urls[self.model_name]['labels']

	@property
	def all_models_available(self):
		return ['NIH', 'RSNA', 'PC', 'CheX', 'MIMIC_NB', 'MIMIC_CH', 'ALL_224', 'ALL_512', 'baseline_jfhealthcare', 'baseline_CheX']

	@classmethod
	def extract_feature_maps(cls, config, data_mode=DataModes.TEST): # type: (argparse.Namespace, str) -> Tuple[np.ndarray, pd.DataFrame]

		LM = cls(config)
		model = LM.load()

		LD = LoadChestXrayDatasets(config=config, pathologies_in_model=model.pathologies)
		LD.load()
		data = getattr(LD, data_mode)

		def process_one_batch(batch_data):

			# Getting the data and its corresponding true labels
			device = 'cuda' if USE_CUDA else 'cpu'
			images = batch_data["img" ].to(device)

			# Get feature maps
			feats = model.features2(images) if hasattr(model, "features2") else model.features(images)
			return feats.reshape(len(feats),-1).detach().cpu(), batch_data["lab" ].to(device).detach().cpu()

		def looping_over_all_batches(data_loader, n_batches_to_process) -> Tuple[np.ndarray, np.ndarray]:

			d_features, d_truth  = [], []
			for batch_idx, batch_data in enumerate(tqdm(data_loader)):
				if n_batches_to_process and (batch_idx >= n_batches_to_process):
					break
				features, truth = process_one_batch(batch_data)
				d_features.append(features)
				d_truth.append(truth)

			return np.concatenate(d_features), np.concatenate(d_truth)

		with torch.no_grad(): # inference_mode no_grad
			feature_maps, truth = looping_over_all_batches(data_loader=data.data_loader, n_batches_to_process=config.n_batches_to_process)

		return feature_maps , pd.DataFrame(truth, columns=model.pathologies), data.labels.nodes.not_null

# endregion


# region Main
class CalculateOriginalFindings:

	def __init__(self, data, model, config, criterion=torch.nn.BCELoss(reduction='none')): # type: (Data, torch.nn.Module, argparse.Namespace, torch.nn.Module) -> None
		self.device 			  = 'cuda' if USE_CUDA else 'cpu'
		self.criterion            = criterion
		self.n_batches_to_process = config   .n_batches_to_process
		self.model                = model
		self.config               = config
		self.save_path_full       = f'details/{config.MLFlow_run_name}/baseline/findings_original_{data.data_mode}.pkl'

		data.initialize_findings(pathologies=self.model.pathologies, experiment_stage=ExperimentSTAGE.ORIGINAL)
		self.data = data

	@staticmethod
	def calculate(data, model, n_batches_to_process, device='cpu', criterion=torch.nn.BCELoss(reduction='none')): # type: (Data, torch.nn.Module, int, str, torch.nn.Module) -> Data

		model.eval()

		def looping_over_all_batches() -> None:

			def process_one_batch(batch_data_in):

				def get_truth_and_predictions():

					# Sample data and its corresponding labels
					images = batch_data_in["img"].to(device)
					truth  = batch_data_in["lab"].to(device)

					# Feeding the samples into the model
					logit = model(images)
					pred  = torch.sigmoid(logit) # .detach().numpy()

					return pred , logit , truth

				def get_loss_per_sample_batch(pred_batch, truth_batch, index):

					l_batch = pd.DataFrame(columns=model.pathologies, index=index)

					for ix, lbl_name in enumerate(model.pathologies):

						# This skips the empty labels (i.e. the labels that are not in both the dataset & model)
						if len(lbl_name) == 0: continue

						task_truth, task_pred = truth_batch[:, ix].double() , pred_batch[:, ix].double()

						# Calculating the loss per sample
						l_batch[lbl_name] = criterion(task_pred, task_truth).detach().cpu().numpy()

					return l_batch


				# Getting the true and predicted labels
				pred_batch, logit_batch, truth_batch = get_truth_and_predictions()

				# Getting the index of the samples for this batch
				index = batch_data_in['idx'].detach().cpu().numpy()

				loss_per_sample_batch = get_loss_per_sample_batch(pred_batch, truth_batch, index)

				# Converting the outputs and targets to dataframes
				# to_df = lambda data: pd.DataFrame(data.detach().cpu().numpy(), columns=model.pathologies, index=index)
				to_df = lambda data: pd.DataFrame(data.detach().cpu(), columns=model.pathologies, index=index)

				return to_df(pred_batch), to_df(logit_batch), to_df(truth_batch), loss_per_sample_batch

			for batch_idx, batch_data in enumerate(data.data_loader):

				# This ensures that  we only evaluate the data for a few batches. End the loop after n_batches_to_process
				if n_batches_to_process and (batch_idx >= n_batches_to_process): break

				pred, logit, truth, loss = process_one_batch(batch_data)

				# Appending the results for this batch to the results
				data.ORIGINAL.pred  = pd.concat([data.ORIGINAL.pred , pred ])
				data.ORIGINAL.logit = pd.concat([data.ORIGINAL.logit, logit])
				data.ORIGINAL.truth = pd.concat([data.ORIGINAL.truth, truth])
				data.ORIGINAL.loss  = pd.concat([data.ORIGINAL.loss , loss ])

		with torch.no_grad():

			# Looping over all batches
			looping_over_all_batches()

			# Measuring AUCs & Thresholds
			data.ORIGINAL = AIM1_1_TorchXrayVision.calculating_Threshold_and_Metrics(data.ORIGINAL)

		data.ORIGINAL.results = { key: getattr( data.ORIGINAL, key ) for key in ['metrics', 'pred', 'logit', 'truth', 'loss'] }
		return data

	def do_calculate(self):

		# Calculating the ORIGINAL findings
		params = {key: getattr(self, key) for key in ['data', 'model', 'device' , 'criterion' , 'n_batches_to_process']}
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
			OG.data.ORIGINAL.results = LoadSaveFindings( config, OG.save_path_full ).load( source=config.do_findings_original, run_name=config.MLFlow_run_name )

			# Adding the ORIGINAL findings to the graph nodes
			OG.data.Hierarchy_cls.update_graph( findings_original=OG.data.ORIGINAL.results )

		return OG.data


class CalculateNewFindings:
	def __init__(self, model, data, hyperparameters, config, approach): # type: (torch.nn.Module, Data, dict, argparse.Namespace, str) -> None

		self.model            = model
		self.hyperparameters  = hyperparameters
		data.initialize_findings(pathologies=self.model.pathologies, experiment_stage=ExperimentSTAGE.NEW)
		self.data             = data
		config.approach       = approach
		self.config           = config
		self.save_path_full   = f'details/{config.MLFlow_run_name}/{approach}/findings_new_{data.data_mode}.pkl'

	@staticmethod
	def get_hierarchy_penalty_for_node(G, node, thresh_technique, parent_condition_mode, approach, a, b=0) -> np.ndarray:
		""" Return:   hierarchy_penalty: Array(index=sample_indices) """

		def calculate_H(parent_node: str) -> np.ndarray:
			""" Return:   hierarchy_penalty: Array(index=sample_indices) """

			def calculate_raw_weight(pdata) -> pd.Series:
				if   approach in ['1', 'logit']: return pd.Series(a * pdata.data.logit.to_numpy()    , index=pdata.data.index)
				elif approach in ['2', 'loss']:  return pd.Series(a * pdata.data.loss.to_numpy() + b , index=pdata.data.index)
				else: raise ValueError(' approach is not supproted')

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
			pdata = Hierarchy.get_findings_for_node(G=G, node=parent_node, WHICH_RESULTS=ExperimentSTAGE.ORIGINAL, thresh_technique=thresh_technique)

			# Calculating the initial hierarchy_penalty based on "a" , "b" and "approach"
			hierarchy_penalty = calculate_raw_weight(pdata)

			# Cleaning up the hierarchy_penalty for the current node: Setting the hierarchy_penalty to 1 for samples where the parent class exist, and Nan if the parent label is Nan
			return apply_parent_doesnot_exist_condition( hierarchy_penalty=hierarchy_penalty, pdata=pdata )

		def set_H_to_be_ineffective() -> np.ndarray:

			ndata = Hierarchy.get_findings_for_node(G=G , node=node, thresh_technique=thresh_technique , WHICH_RESULTS=ExperimentSTAGE.ORIGINAL)

			if   approach in ['1', 'logit']: return np.zeros(len(ndata.data.index))
			elif approach in ['2', 'loss' ]: return np.ones(len(ndata.data.index))

			raise ValueError(' approach is not supproted')

		# Get the parent node of the current node. We assume that each node can only have one parent to aviod complications in theoretical calculations.
		parent_node = Hierarchy.get_parent_node(G=G, node=node)

		# Calculating the hierarchy_penalty for the current node
		return calculate_H(parent_node) if parent_node else set_H_to_be_ineffective()

	@staticmethod
	def do_approach(config, w, ndata): # type: (argparse.Namespace, pd.Series, Hierarchy.OUTPUT) -> Tuple[np.ndarray, np.ndarray]

		def do_approach2_per_node():

			def calculate_loss_gradient(p, y): # type: (np.ndarray, np.ndarray) -> pd.Series
				return -y / (p + 1e-7) + (1 - y) / (1 - p + 1e-7)

			def update_pred(l_new, l_gradient): # type: (pd.Series, pd.Series) -> pd.Series
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


		if  config.approach in ('1', 'logit'):
			pred_new, logit_new = do_approach1_per_node()
			loss_new = np.ones(pred_new.shape) * np.nan

		elif config.approach in ('2', 'loss') :
			pred_new, loss_new  = do_approach2_per_node()
			logit_new = np.ones(pred_new.shape) * np.nan

		return pred_new, logit_new, loss_new

	@staticmethod
	def calculate_per_node(node, data, config, hyperparameters, thresh_technique): # type: (str, Data, argparse.Namespace, Dict[str, pd.DataFrame], str) -> Data

		x = thresh_technique
		G = data.Hierarchy_cls.G

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
		a, b = hyperparameters[x].loc['a',node], hyperparameters[x].loc['b',node]
		data.NEW.hierarchy_penalty[x,node] = CalculateNewFindings.get_hierarchy_penalty_for_node( G=G, a=a, b=b, node=node, thresh_technique=x, parent_condition_mode=config.parent_condition_mode, approach=config.approach )

		# Getting node data
		ndata: Hierarchy.OUTPUT = Hierarchy.get_findings_for_node( G=G, node=node, thresh_technique=x, WHICH_RESULTS=ExperimentSTAGE.ORIGINAL )

		data.NEW.pred[x,node], data.NEW.logit[x,node], data.NEW.loss[x,node] = CalculateNewFindings.do_approach( config=config, w=data.NEW.hierarchy_penalty[x][node], ndata=ndata )

		data.NEW = AIM1_1_TorchXrayVision.calculating_Threshold_and_Metrics_per_node( node=node, findings=data.NEW, thresh_technique=x )

		return data

	@staticmethod
	def calculate(data, config, hyperparameters): # type: (Data, argparse.Namespace, Dict[str, pd.DataFrame]) -> Data

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
			for node in data.labels.nodes.not_null:
				data = CalculateNewFindings.calculate_per_node( node=node, data=data, config=config, hyperparameters=hyperparameters, thresh_technique=x )

		data.NEW.results = { key: getattr( data.NEW, key ) for key in ['metrics', 'pred', 'logit', 'truth', 'loss', 'hierarchy_penalty'] }

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
	def get_updated_data(cls, model, data, hyperparameters, config, approach):  # type: (torch.nn.Module, Data, dict, argparse.Namespace, str) -> Data

		# Initializing the class
		NEW = cls(model=model, data=data, hyperparameters=hyperparameters, config=config, approach=approach)

		if config.do_findings_new == 'calculate':
			NEW.do_calculate()
		else:
			NEW.data.NEW.results = LoadSaveFindings( NEW.config, NEW.save_path_full ).load( source=config.do_findings_new, run_name=config.MLFlow_run_name )

			# Adding the ORIGINAL findings to the graph nodes
			NEW.data.Hierarchy_cls.update_graph( findings_new=NEW.data.NEW.results )

		return NEW.data


class HyperParameterTuning:

	def __init__(self, config, data , model): # type: (argparse.Namespace, Data, torch.nn.Module) -> None

		self.config      = config
		self.data 		 = data
		self.model       = model
		self.save_path_full = f'details/{config.MLFlow_run_name}/{config.approach}/hyperparameters.pkl'
		self.hyperparameters = None

		# Initializing the data.NEW
		if data.NEW is None:
			data.initialize_findings(pathologies=model.pathologies, experiment_stage=ExperimentSTAGE.NEW)

	def initial_hyperparameters(self, a=0.0 , b=1.0): # type: (float, float) -> Dict[str, pd.DataFrame]
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
	def calculate(initial_hp, config, data): # type: (Dict[str, pd.DataFrame], argparse.Namespace, Data) -> Dict[str, pd.DataFrame]

		def extended_calculate_per_node(nt: List[str]) -> List[Union[str, float]]:
			hyperparameters_updated = HyperParameterTuning.calculate_per_node( node=nt[0], thresh_technique=nt[1],data=data, config=config, hyperparameters=deepcopy(initial_hp))
			return [nt[0], nt[1]] + hyperparameters_updated # type: ignore

		def update_hyperparameters(results_in): # type: ( List[Tuple[str, str, List[float]]] ) -> Dict[str, pd.DataFrame]
			# Creating a copy of initial hyperparameters
			hp_in = initial_hp.copy()

			# Updating hyperparameters with new findings
			for r_in in results_in:
				hp_in[r_in[1]][r_in[0]] = r_in[2]

			return hp_in

		stop_requested = False

		# Set the maximum number of open files
		resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))

		# do_PerClass = partial(extended_calculate_per_node(data=data, config=config, hyperparameters=initial_hp))

		PARALLELIZE = config.parallelization_technique

		with contextlib.suppress(KeyboardInterrupt):

			if PARALLELIZE == 0:
				results = [extended_calculate_per_node(node_thresh) for node_thresh in data.labels.nodes.node_thresh_tuple]

			elif PARALLELIZE == 1:
				multiprocessing.set_start_method('spawn', force=True)
				with multiprocessing.Pool(processes=4) as pool: # Set the number of worker processes here
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
			HP.hyperparameters = LoadSaveFindings(config, HP.save_path_full).load(source=config.do_hyperparameters, run_name=config.MLFlow_run_name)

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


@dataclass(init=False)
class DataMerged:
	def __init__(self, data):
		self.pred  = pd.concat(data['pred'],axis=0).reset_index(drop=True)
		self.truth = pd.concat(data['truth'],axis=0).reset_index(drop=True)
		self.yhat  = pd.concat(data['yhat'], axis=0).reset_index(drop=True).astype(int)
		self.list_nodes_impacted = [n for n in self.pred.columns if n in set().union(*data['list_nodes_impacted'])]
		self.auc_acc_f1 		 = pd.DataFrame(columns=self.pred.columns, index=['AUC', 'ACC', 'F1'])

@dataclass
class Metrics:
	metrics_comparison:  pd.DataFrame
	config  : argparse.Namespace
	approach: str
	baseline: DataMerged
	proposed: DataMerged

	@property
	def metrics(self):
		return self.metrics_comparison[self.baseline.list_nodes_impacted].T


@dataclass
class MetricsAllTechniques:
	loss: Metrics
	logit: Metrics
	auc_acc_f1: pd.DataFrame
	thresh_technique: str
	datasets_list: List[str]
	data_mode: str

	@property
	def metrics(self):
		return self.auc_acc_f1.T[self.list_nodes_impacted].T.astype(float).round(3)

	@property
	def config(self):
		return self.loss.config

	@property
	def list_nodes_impacted(self):
		return self.logit.baseline.list_nodes_impacted

	def plot_metrics(self, save_figure=True, figsize=(21, 7), font_scale=1.8, fontsize=20):

		def save_plot():
			save_path = self.config.local_path.joinpath(f'figures/auc_acc_f1_all_datasets/{self.thresh_technique}/')
			save_path.mkdir(parents=True, exist_ok=True)
			for format in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath( f'metrics_AUC_ACC_F1.{format}'), format=format, dpi=300)

		def barplot():
			fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True, rc=None)

			params = dict(legend=False, fontsize=16, kind='barh')

			self.metrics['ACC'].plot(ax=axes[0], title='ACC', **params)
			self.metrics['AUC'].plot(ax=axes[1], title='AUC', **params)
			self.metrics['F1'].plot(ax=axes[2], title='F1', **params)
			plt.legend(loc='upper right', fontsize=16)
			plt.tight_layout()


		def heatmap():
			import seaborn as sns
			import matplotlib.pyplot as plt

			sns.set(font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper', color_codes=True, rc=None)

			fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
			params = dict(annot=True, fmt=".3f", linewidths=.5, cmap='YlGnBu', cbar=False, annot_kws={"size": fontsize})

			for i, m in enumerate(['ACC', 'AUC', 'F1']):

				sns.heatmap(data=self.metrics[m],ax=axes[i], **params)
				axes[i].set_title(m, fontsize=int(1.5*fontsize), fontweight='bold')
				axes[i].tick_params(axis='both', which='major', labelsize=fontsize)

				# sns.heatmap(data=self.metrics['ACC'],ax=axes[0], **params)
				# axes[0].set_title('ACC', fontsize=int(1.5*fontsize), fontweight='bold')
				# axes[0].tick_params(axis='both', which='major', labelsize=fontsize)

				# sns.heatmap(data=self.metrics['AUC'], ax=axes[1], **params)
				# axes[1].set_title('AUC', fontsize=int(1.5*fontsize), fontweight='bold')
				# axes[1].tick_params(axis='both', which='major', labelsize=fontsize)

				# sns.heatmap(data=self.metrics['F1'] , ax=axes[2], **params)
				# axes[2].set_title('F1', fontsize=int(1.5*fontsize), fontweight='bold')
				# axes[2].tick_params(axis='both', which='major', labelsize=fontsize)

			plt.tight_layout()

		heatmap()

		if save_figure:
			save_plot()

	def plot_roc_curves(self, save_figure=True, figsize=(15, 15), font_scale=1.8, fontsize=20, labelpad=0):

		def save_plot():
			save_path = self.config.local_path.joinpath( f'figures/roc_curve_all_datasets/{self.thresh_technique}/')
			save_path.mkdir(parents=True, exist_ok=True)
			for format in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(
					f'roc_curve_all_datasets.{format}'), format=format, dpi=300)

		# Set up the grid
		def setup_plot():

			# Set a seaborn style for visually appealing plots
			sns.set(font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper', color_codes=True, rc=None)


			# Set up the grid
			n_nodes, n_cols = len(self.list_nodes_impacted), 3
			n_rows = int(np.ceil(n_nodes / n_cols))

			# Set up the figure and axis
			fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True)
			axes     = axes.flatten()


			# sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True, rc=None)

			return fig, axes, n_rows, n_cols

		list_parent_nodes = set(Hierarchy.taxonomy_structure.keys()).intersection( set(self.logit.proposed.pred.columns))

		fig, axes, n_rows, n_cols = setup_plot()

		def plot_per_node(node, idx):

			row_idx = idx // n_cols
			col_idx = idx % n_cols
			ax      = axes[idx]

			# Calculate the ROC curve and AUC
			def get_fpr_tpr_auc(pred_node, truth_node, technique, roc_auc):

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

			truth = self.logit.proposed.truth
			for technique, data in [('baseline', self.logit.baseline), ('logit', self.logit.proposed), ('loss', self.loss.proposed)]:
				line = get_fpr_tpr_auc( pred_node=data.pred[node], truth_node=truth[node], technique=technique, roc_auc=data.auc_acc_f1[node]['AUC'])
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
		for idx, node in enumerate(self.list_nodes_impacted):
			plot_per_node(node, idx)

		# Postprocess the plot
		postprocess()

		# Save the plot
		if save_figure:
			save_plot()


@dataclass
class MetricsAllTechniquesThresholds:
	DEFAULT: 		  MetricsAllTechniques
	ROC: 			  MetricsAllTechniques
	PRECISION_RECALL: MetricsAllTechniques


class AIM1_1_TorchXrayVision():

	def __init__(self, config: argparse.Namespace, seed: int=10):

		self.hyperparameters = None
		self.config         : argparse.Namespace                  = config
		self.train          : Optional[Data]                      = None
		self.test           : Optional[Data]                      = None
		self.model          : Optional[torch.nn.Module]           = None
		self.d_data         : Optional[xrv.datasets.CheX_Dataset] = None

		approach = config.approach if hasattr(config, 'approach') else 'loss'
		self.save_path : str = f'details/{config.MLFlow_run_name}/{approach}'

		# Setting the seed
		self.setting_random_seeds_for_PyTorch(seed=seed)


	@staticmethod
	def measuring_BCE_loss(p, y):
		return -( y * np.log(p) + (1 - y) * np.log(1 - p) )

	@staticmethod
	def equations_sigmoidprime(p):
		""" Refer to Eq. (10) in the paper draft """
		return p*(1-p)

	@staticmethod
	def setting_random_seeds_for_PyTorch(seed=10):
		np.random.seed(seed)
		torch.manual_seed(seed)
		if USE_CUDA:
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark     = False

	def threshold(self, data_mode = DataModes.TRAIN):
		
		data = self.train if data_mode == DataModes.TRAIN else self.test

		exp_stage_list   = [ExperimentSTAGE.ORIGINAL.name       , ExperimentSTAGE.NEW.name]
		thresh_tech_list = [ThreshTechList.PRECISION_RECALL.name, ThreshTechList.ROC.name]
		
		df = pd.DataFrame(  index=data.ORIGINAL.threshold.index ,
							columns=pd.MultiIndex.from_product([thresh_tech_list, exp_stage_list]) )

		for th_tqn in [ThreshTechList.ROC.value , ThreshTechList.PRECISION_RECALL.value]:
			df[ (th_tqn,ExperimentSTAGE.ORIGINAL.name) ] = data.ORIGINAL.threshold[th_tqn]
			df[ (th_tqn,ExperimentSTAGE.NEW.name     ) ] = data.NEW     .threshold[th_tqn]

		return df.replace(np.nan, '')

	@staticmethod
	def accuracy_per_node(data, node, WHICH_RESULTS, thresh_technique) -> float:

		findings = getattr(data,WHICH_RESULTS)

		if node in data.labels.nodes.not_null:
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
				df.loc[node, xf] = AIM1_1_TorchXrayVision.accuracy_per_node(data=data, node=node, thresh_technique=xf[0], WHICH_RESULTS=xf[1])

		return df.replace(np.nan, '')

	def findings_per_node(self, node, data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		# Getting the hierarchy_penalty for node
		hierarchy_penalty = pd.DataFrame(columns=ThreshTechList)
		for x in ThreshTechList:
			hierarchy_penalty[x] = data.hierarchy_penalty[x,node]

		# Getting Metrics for node
		metrics = pd.DataFrame()
		for m in ['AUC','ACC','F1']:
			metrics[m] = self.get_metric(metric=m, data_mode=data_mode).T[node]

		return Hierarchy.OUTPUT(hierarchy_penalty=hierarchy_penalty, metrics=metrics, data=data)

	def findings_per_node_iterator(self, data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		return iter( [ self.findings_per_node(node)  for node in data.Hierarchy_cls.parent_dict.keys() ] )

	def findings_per_node_with_respect_to_their_parent(self, node, thresh_technq = 'ROC', data_mode=DataModes.TRAIN):

		data = self.train if data_mode == DataModes.TRAIN else self.test

		N = data.Hierarchy_cls.G.nodes
		parent_child = data.Hierarchy_cls.parent_dict[node] + [node]

		df = pd.DataFrame(index=N[node][ ExperimentSTAGE.ORIGINAL ][ 'data' ].index , columns=pd.MultiIndex.from_product(  [  parent_child,['truth' , 'pred' , 'loss'],[ExperimentSTAGE.ORIGINAL,ExperimentSTAGE.NEW]  ] ))

		for n in parent_child:
			for dtype in ['truth' , 'pred' , 'loss']:
				df[ (n, dtype, ExperimentSTAGE.ORIGINAL)] = N[n][ ExperimentSTAGE.ORIGINAL ][ 'data' ][dtype].values
				df[ (n , dtype, ExperimentSTAGE.NEW)]     = N[n][ ExperimentSTAGE.NEW ][ 'data' ][thresh_technq][dtype].values

			df[(n, 'hierarchy_penalty', ExperimentSTAGE.NEW)] = N[n][ 'hierarchy_penalty' ][thresh_technq].values

		return df.round(decimals=3).replace(np.nan, '', regex=True)

	@staticmethod
	def calculating_Threshold_and_Metrics(DATA): # type: (Findings) -> Findings
		for x in ThreshTechList:
			for node in DATA.pathologies:
				DATA = AIM1_1_TorchXrayVision.calculating_Threshold_and_Metrics_per_node( node=node, findings=DATA, thresh_technique=x )

		return DATA

	@staticmethod
	def calculating_Threshold_and_Metrics_per_node(node, findings, thresh_technique):  # type: (str, Findings, str) -> Findings

		def calculating_optimal_Thresholds(y, yhat, x):

			if x == 'DEFAULT':
				findings.metrics[x,node]['Threshold'] = 0.5

			if x == 'ROC':
				fpr, tpr, th = sklearn.metrics.roc_curve(y, yhat)
				findings.metrics[x,node]['Threshold'] = th[np.argmax( tpr - fpr )]

			if x == 'PRECISION_RECALL':
				ppv, recall, th = sklearn.metrics.precision_recall_curve(y, yhat)
				f_score = 2 * (ppv * recall) / (ppv + recall)
				findings.metrics[x,node]['Threshold'] = th[np.argmax( f_score )]

		def calculating_Metrics(y, yhat, x):
			findings.metrics[x, node]['AUC'] = sklearn.metrics.roc_auc_score( y, yhat )
			findings.metrics[x, node]['ACC'] = sklearn.metrics.accuracy_score( y, yhat >= findings.metrics[x, node]['Threshold'] )
			findings.metrics[x, node]['F1']  = sklearn.metrics.f1_score      ( y, yhat >= findings.metrics[x, node]['Threshold'] )

		# Finding the indices where the truth is not nan
		non_null = ~np.isnan( findings.truth[node] )
		truth_notnull = findings.truth[node][non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
			# for thresh_technique in ThreshTechList:
			pred = findings.pred[node] if findings.experiment_stage == ExperimentSTAGE.ORIGINAL else findings.pred[thresh_technique,node]
			pred_notnull = pred[non_null].to_numpy()

			calculating_optimal_Thresholds( y = truth_notnull, yhat = pred_notnull, x = thresh_technique )
			calculating_Metrics( y            = truth_notnull, yhat = pred_notnull, x = thresh_technique )

		return findings

	def save_metrics(self):

		for metric in ['AUC', 'ACC', 'F1', 'Threshold']:

			# Saving the data
			path = self.config.local_path.joinpath( f'{self.save_path}/{metric}.xlsx' )

			# Create a new Excel writer
			with pd.ExcelWriter(path, engine='openpyxl') as writer:

				# Loop through the data modes
				for data_mode in DataModes.members():
					self.get_metric(metric=metric, data_mode=data_mode).to_excel(writer, sheet_name=data_mode.value)

				# Save the Excel file
				# writer.save()



	def get_metric(self, metric='AUC', data_mode=DataModes.TRAIN) -> pd.DataFrame:

		data: Data = self.train if data_mode == DataModes.TRAIN else self.test

		column_names = data.labels.nodes.impacted

		columns = pd.MultiIndex.from_product([ThreshTechList, [ ExperimentSTAGE.ORIGINAL, ExperimentSTAGE.NEW]], names=['thresh_technique', 'WR'])
		df = pd.DataFrame(index=data.ORIGINAL.pathologies, columns=columns)

		for x in ThreshTechList:
			if hasattr(data.ORIGINAL, 'metrics'):
				df[x, ExperimentSTAGE.ORIGINAL] = data.ORIGINAL.metrics[x].T[metric]
			if hasattr(data.NEW, 'metrics'):
				df[x, ExperimentSTAGE.NEW] = data.NEW     .metrics[x].T[metric]

		df = df.apply(pd.to_numeric, errors='ignore').round(3).replace(np.nan, '')

		return df.T[column_names].T


	@staticmethod
	def get_data_and_model(config):

		# Load the model
		model = LoadModelXRV(config).load()

		# Load the data
		LD = LoadChestXrayDatasets(config=config, pathologies_in_model=model.pathologies)
		LD.load()

		return LD.train, LD.test, model, LD.d_data

	@classmethod
	def run_full_experiment(cls, approach='loss', seed=10, **kwargs):

		# Getting the user arguments
		config = reading_user_input_arguments(jupyter=True, approach=approach, **kwargs)

		# Initializing the class
		FE = cls(config=config, seed=seed)

		# Loading train/test data as well as the pre-trained model
		FE.train, FE.test, FE.model, FE.d_data = cls.get_data_and_model(FE.config)

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
		FE.train = CalculateNewFindings.get_updated_data(data=FE.train, approach=approach, **param_dict)
		FE.test  = CalculateNewFindings.get_updated_data(data=FE.test , approach=approach, **param_dict)

		if FE.config.RUN_MLFlow and FE.config.KILL_MLFlow_at_END:
			FE.kill_MLFlow()

		# Saving the metrics: AUC, threshold, accuracy
		FE.save_metrics()

		return FE

	@staticmethod
	def loop_run_full_experiment():
		for dataset_name in DatasetList:
			for approach in ['logit', 'loss']:
				AIM1_1_TorchXrayVision.run_full_experiment(approach=approach, dataset_name=dataset_name.name)

	@classmethod
	def get_merged_data(cls, data_mode=DataModes.TEST, approach='logit', thresh_technique='DEFAULT', datasets_list=None): # type: (str, str, str, list) -> Tuple[DataMerged, DataMerged]

		if datasets_list is None:
			datasets_list = DatasetList
			
		def get(method: ExperimentSTAGE) -> DataMerged:
			data = defaultdict(list)
			for dataset_name in datasets_list:
				a1 = cls.run_full_experiment(approach=approach, dataset_name=dataset_name)

				metric = getattr( getattr(a1,data_mode),method.value)
				data['pred'].append(metric.pred[thresh_technique] if method==ExperimentSTAGE.NEW else metric.pred)
				data['truth'].append(metric.truth)
				data['yhat'].append(data['pred'][-1] >= metric.metrics[thresh_technique].T['Threshold'].T )
				data['list_nodes_impacted'].append(getattr(a1, data_mode).labels.nodes.impacted)

			return DataMerged(data)

		baseline = get(ExperimentSTAGE.ORIGINAL)
		proposed = get(ExperimentSTAGE.NEW)

		return baseline, proposed

	@classmethod
	def get_all_metrics(cls, datasets_list=['CheX', 'NIH', 'PC'], data_mode=DataModes.TEST, thresh_technique='DEFAULT', jupyter=True, **kwargs): # type: (list, str, str, bool, dict) -> MetricsAllTechniques

		config = reading_user_input_arguments(jupyter=jupyter, **kwargs)
		save_path = pathlib.Path(f'tables/metrics_all_datasets/{thresh_technique}')

		def apply_to_approach(approach): # type: (str) -> Metrics

			baseline, proposed = cls.get_merged_data(data_mode=data_mode, approach=approach, thresh_technique=thresh_technique, datasets_list=datasets_list)

			def get_AUC_ACC_F1(node: str, data: DataMerged):

				# Finding the indices where the truth is not nan
				non_null = ~np.isnan( data.truth[node] )
				truth_notnull = data.truth[node][non_null].to_numpy()

				if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
					data.auc_acc_f1[node]['AUC'] = sklearn.metrics.roc_auc_score( data.truth[node][non_null], data.yhat[node][non_null])
					data.auc_acc_f1[node]['ACC'] = sklearn.metrics.accuracy_score(data.truth[node][non_null], data.yhat[node][non_null])
					data.auc_acc_f1[node]['F1']  = sklearn.metrics.f1_score( 	 data.truth[node][non_null], data.yhat[node][non_null])

			def get_p_value_kappa_cohen_d_BF10(df, node): # type: (pd.DataFrame, str) -> None

				# Perform the independent samples t-test
				df.loc['t_stat',node], df.loc['p_value',node] = stats.ttest_ind( baseline.yhat[node], proposed.yhat[node])

				# kappa inter rater metric
				df.loc['kappa',node] = sklearn.metrics.cohen_kappa_score(baseline.yhat[node], proposed.yhat[node])

				df_ttest = pg.ttest(baseline.yhat[node], proposed.yhat[node])
				df.loc['power',node]   = df_ttest['power'].values[0]
				df.loc['cohen-d',node] = df_ttest['cohen-d'].values[0]
				df.loc['BF10',node]    = df_ttest['BF10'].values[0]

			metrics_comparison = pd.DataFrame(columns=baseline.pred.columns, index=['kappa', 'p_value', 't_stat', 'power', 'cohen-d','BF10'])
			# auc_acc_f1_baseline = pd.DataFrame(columns=baseline.pred.columns, index=['AUC', 'ACC', 'F1'])
			# auc_acc_f1_proposed = pd.DataFrame(columns=baseline.pred.columns, index=['AUC', 'ACC', 'F1'])

			for node in baseline.pred.columns:
				get_AUC_ACC_F1(node, baseline)
				get_AUC_ACC_F1(node, proposed)
				get_p_value_kappa_cohen_d_BF10(metrics_comparison, node)

			return Metrics(metrics_comparison=metrics_comparison, baseline=baseline, proposed=proposed, config=config, approach=approach)

		def get_AUC_ACC_F1_merged(logit, loss): # type: (Metrics, Metrics) -> pd.DataFrame
			columns = pd.MultiIndex.from_product( [['ACC', 'AUC', 'F1'], ['baseline', 'loss', 'logit']])
			auc_acc_f1 = pd.DataFrame(columns=columns)

			for metric in ['ACC', 'AUC', 'F1']:
				auc_acc_f1[metric] = pd.DataFrame( dict(baseline=loss.baseline.auc_acc_f1.T[metric],
														loss=loss.proposed.auc_acc_f1.T[metric],
														logit=logit.proposed.auc_acc_f1.T[metric] ))

			return auc_acc_f1

		if config.do_metrics == 'calculate':
			logit 	   = apply_to_approach('logit')
			loss  	   = apply_to_approach('loss')
			auc_acc_f1 = get_AUC_ACC_F1_merged(logit, loss)

			# Saving the metrics locally
			LoadSaveFindings(config, save_path.joinpath('logit_metrics.csv')).save(logit.metrics_comparison[logit.baseline.list_nodes_impacted].T)
			LoadSaveFindings(config, save_path.joinpath('logit.pkl')).save(logit)

			LoadSaveFindings(config, save_path.joinpath('loss_metrics.csv')).save(loss.metrics_comparison[loss.baseline.list_nodes_impacted].T)
			LoadSaveFindings(config, save_path.joinpath('loss.pkl')).save(loss)

			LoadSaveFindings(config, save_path.joinpath(f'auc_acc_f1.xlsx')).save(auc_acc_f1, index=True)

		else:
			load_lambda = lambda x, **kwargs: LoadSaveFindings(config, save_path.joinpath(x)).load(source=config.do_metrics, run_name=config.MLFlow_run_name, **kwargs)
			logit 	   = load_lambda('logit.pkl')
			loss 	   = load_lambda('loss.pkl')
			auc_acc_f1 = load_lambda('auc_acc_f1.xlsx', index_col=0, header=[0, 1])

		return MetricsAllTechniques(loss=loss, logit=logit, auc_acc_f1=auc_acc_f1, thresh_technique=thresh_technique, datasets_list=datasets_list, data_mode=data_mode)

	@classmethod
	def get_all_metrics_all_thresh_techniques(cls, datasets_list: list[str]=['CheX', 'NIH', 'PC'], data_mode: str=DataModes.TEST) -> MetricsAllTechniquesThresholds:

		output = {}
		for x in tqdm(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
			output[x] = AIM1_1_TorchXrayVision.get_all_metrics(datasets_list=datasets_list, data_mode=data_mode, thresh_technique=x)

		return MetricsAllTechniquesThresholds(**output)

	@staticmethod
	def check(check_what='labels', jupyter=True, **kwargs):

		config = reading_user_input_arguments(jupyter=jupyter)

		if check_what == 'config':
			return config

		elif check_what == 'model':
			LM = LoadModelXRV(config)
			LM.load()
			return LM

		elif check_what in ('DT', 'labels', 'd_data', 'show_graph'):
			model = LoadModelXRV(config).load()
			DT = LoadChestXrayDatasets(config=config, pathologies_in_model=model.pathologies)
			DT.load()

			if   check_what == 'DT':	     return DT
			elif check_what == 'labels':     return getattr(DT, kwargs['data_mode']).labels
			elif check_what == 'd_data': 	 return DT.d_data
			elif check_what == 'show_graph': DT.train.Hierarchy_cls.show(package=kwargs['package'])

	def visualize(self, data_mode=DataModes.TEST, thresh_technique='DEFAULT', **kwargs):
		return Visualize(data=getattr(self, data_mode.value), thresh_technique=thresh_technique, config=self.config, **kwargs)

# endregion


# region Other

def reading_user_input_arguments(argv=None, jupyter=True, config_name='config.json', **kwargs) -> argparse.Namespace:

	def parse_args() -> argparse.Namespace:
		"""	Getting the arguments from the command line
			Problem: 	Jupyter Notebook automatically passes some command-line arguments to the kernel.
						When we run argparse.ArgumentParser.parse_args(), it tries to parse those arguments, which are not recognized by your argument parser.
			Solution: 	To avoid this issue, you can modify your get_args() function to accept an optional list of command-line arguments, instead of always using sys.argv.
						When this list is provided, the function will parse the arguments from it instead of the command-line arguments. """

		# If argv is not provided, use sys.argv[1:] to skip the script name
		args = [] if jupyter else (argv or sys.argv[1:])

		args_list = [
					# Dataset
					dict(name = 'dataset_name', type = str, help = 'Name of the dataset'               ),
					dict(name = 'data_mode'   , type = str, help = 'Dataset mode: train or valid'      ),
					dict(name = 'max_sample'  , type = int, help = 'Maximum number of samples to load' ),

					# Model
					dict(name='model_name'   , type=str , help='Name of the pre_trained model.' ),
					dict(name='architecture' , type=str , help='Name of the architecture'       ),

					# Training
					dict(name = 'batch_size'     , type = int   , help = 'Number of batches to process' ),
					dict(name = 'n_epochs'       , type = int   , help = 'Number of epochs to process'  ),
					dict(name = 'learning_rate'  , type = float , help = 'Learning rate'                ),
					dict(name = 'n_augmentation' , type = int   , help = 'Number of augmentations'      ),

					# Hyperparameter Optimization
					dict(name = 'parent_condition_mode', type = str, help = 'Parent condition mode: truth or predicted' ),
					dict(name = 'approach'             , type = str, help = 'Hyper parameter optimization approach' ),
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

	def updating_config_with_kwargs(updated_args, kwargs):
		if kwargs and len(kwargs) > 0:
			for key in kwargs.keys():
				updated_args[key] = kwargs[key]
		return updated_args

	def get_config(args): # type: (argparse.Namespace) -> argparse.Namespace

		# Loading the config.json file
		config_dir =  os.path.join(os.path.dirname(__file__), config_name if jupyter else args.config)

		if os.path.exists(config_dir):
			with open(config_dir) as f:
				config_raw = json.load(f)

			# converting args to dictionary
			args_dict = vars(args) if args else {}

			# Updating the config with the arguments as command line input
			updated_args ={key: args_dict.get(key) or values for key, values in config_raw.items() }

			# Updating the config with the arguments as function input: used for facilitating the jupyter notebook access
			updated_args = updating_config_with_kwargs(updated_args, kwargs)

			# Convert the dictionary to a Namespace
			args = argparse.Namespace(**updated_args)

			# Updating the paths to their absolute path
			args.local_path 				= pathlib.Path(__file__).parent.parent.parent.parent.joinpath(args.local_path)
			args.dataset_path 			    = pathlib.Path(__file__).parent.parent.parent.parent.joinpath(args.dataset_path)
			args.path_baseline_CheX_weights = pathlib.Path(__file__).parent.parent.parent.parent.joinpath(args.path_baseline_CheX_weights)
			args.MLFlow_run_name 			= f'{args.dataset_name}-{args.model_name}'

		return args

	# Updating the config file
	return  get_config(args=parse_args())


class Tables:

	def __init__(self, jupyter=True, **kwargs):
		self.config = reading_user_input_arguments(jupyter=jupyter, **kwargs)

	def get_metrics_per_thresh_techniques(self, save_table=True, data_mode=DataModes.TEST, thresh_technique='DEFAULT'):

		save_path = self.config.local_path.joinpath(f'tables/metrics_per_dataset/{thresh_technique}/metrics_{data_mode}.xlsx')

		def save(metrics):

			save_path.parent.mkdir(parents=True, exist_ok=True)

			# Create a new Excel writer
			with pd.ExcelWriter(save_path, engine='openpyxl') as writer:

				# Write each metric to a different worksheet
				for m in ['auc', 'f1', 'acc']:
					getattr(metrics, m).to_excel(writer, sheet_name=m.upper())

		def get():
			columns = pd.MultiIndex.from_product([DatasetList.members(),['baseline', 'on_logit', 'on_loss']], names=['dataset', 'approach'])
			auc = pd.DataFrame(columns=columns)
			f1  = pd.DataFrame(columns=columns)
			acc = pd.DataFrame(columns=columns)

			for dataset_name in ['CheX','NIH','PC']:

				df = AIM1_1_TorchXrayVision.run_full_experiment( approach='logit', dataset_name=dataset_name)
				auc [ (dataset_name, 'on_logit') ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['AUC' ]
				f1  [ (dataset_name, 'on_logit') ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['F1'  ]
				acc [ (dataset_name, 'on_logit') ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['ACC' ]

				auc [ (dataset_name, 'baseline') ] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc['AUC' ]
				f1  [ (dataset_name, 'baseline') ] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc['F1'  ]
				acc [ (dataset_name, 'baseline') ] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc['ACC' ]


				df = AIM1_1_TorchXrayVision.run_full_experiment( approach='loss', dataset_name=dataset_name)
				auc [ (dataset_name, 'on_loss' ) ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['AUC' ]
				f1  [ (dataset_name, 'on_loss' ) ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['F1'  ]
				acc [ (dataset_name, 'on_loss' ) ] = getattr(df, data_mode).NEW     .metrics[thresh_technique].loc['ACC' ]


			auc = auc.apply(pd.to_numeric).round(3).replace(np.nan,'')
			f1  = f1 .apply(pd.to_numeric).round(3).replace(np.nan,'')
			acc = acc.apply(pd.to_numeric).round(3).replace(np.nan,'')

			# region load Data & Model
			class Metrics:
				def __init__(self, auc=None, acc=None, f1=None, threshold=None):
					self.auc       = auc
					self.acc       = acc
					self.f1        = f1
					self.threshold = threshold

			return Metrics(auc=auc, f1=f1, acc=acc)

		metrics = get()

		if save_table:
			save(metrics)

		return metrics

	def get_table_datasets_samples(self, save_table=True):

		save_path = pathlib.Path('tables/metrics_all_datasets/table_datasets_samples.csv')

		def get() -> pd.DataFrame:

			def get_PA_AP(mode: str, dname: str) -> pd.Series:

				combine_PA_AP = lambda row: '' if (not row.PA) and (not row.AP) else f"{row.PA}/{row.AP}"

				df2 = pd.DataFrame(columns=['PA', 'AP'])
				for views in ['PA', 'AP']:

					# Getting the dataset for a specific view
					LD = LoadChestXrayDatasets.get_dataset_unfiltered(update_empty_parent_class=(mode=='updated'), dataset_name=dname, views=views)
					df2[views] = LD.labels.sum(axis=0).astype(int).replace(0, '')

					# Adding the Total row
					df2.loc['Total', views] = LD.labels.shape[0]

				return df2.apply(combine_PA_AP, axis=1)


			columns = pd.MultiIndex.from_product( [[ExperimentSTAGE.ORIGINAL, 'updated'], DatasetList.members()])
			df = pd.DataFrame(columns=columns)

			for mode, dname in itertools.product([ExperimentSTAGE.ORIGINAL, 'updated'], DatasetList.members()):
				df[(mode, dname)] = get_PA_AP(mode=mode, dname=dname)

			return df


		df = get()

		if save_table:
			LoadSaveFindings(self.config, save_path).save(df)

		return df


class Visualize:

	def __init__(self, jupyter=True, **kwargs):
		self.config = reading_user_input_arguments(jupyter=jupyter, **kwargs)


	@staticmethod
	def plot_class_relationships(config, method='TSNE', data_mode=DataModes.TEST, feature_maps=None , labels=None): # type: (argparse.Namespace, str, str, Optional[np.ndarray], Optional[Labels]) -> None

		path_main = config.local_path.joinpath(f'{config.MLFlow_run_name}/class_relationship')

		def get_reduced_features(feature_maps, method): # type: (np.ndarray, str) -> np.ndarray
			if method.upper() == 'UMAP':
				reducer = umap.UMAP()
				return reducer.fit_transform(feature_maps)

			elif method.upper() == 'TSNE':
				from sklearn.manifold import TSNE
				return TSNE(n_components=2, random_state=42).fit_transform(feature_maps)

		def do_plot(X_embedded, df_truth, method): # type: (np.ndarray, pd.DataFrame, str) -> None

			def save_plot():

				# output path
				filename = f'class_relationship_{method}'
				path     = path_main.joinpath(method)
				path.mkdir(parents=True, exist_ok=True)

				# Save the plot
				for format in ['png', 'eps', 'svg', 'pdf']:
					plt.savefig(path.joinpath(f'{filename}.{format}'), format=format, dpi=300)

			colors  = plt.cm.get_cmap('tab20',max(18,len(df_truth.columns)))
			_, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
			axes    = axes.flatten()

			for i, node in enumerate(df_truth.columns):
				class_indices = df_truth[df_truth[node].eq(1)].index.to_numpy()
				axes[i].scatter(X_embedded[:, 0], X_embedded[:, 1], color='lightgray', alpha=0.2)
				axes[i].scatter(X_embedded[class_indices, 0], X_embedded[class_indices, 1], c=[colors(i)], alpha=0.5, s=20)
				axes[i].set_title(node)

			plt.suptitle(f"{method} Visualization for {config.dataset_name} dataset")

			# Save the plot
			save_plot()

			plt.show()

		# Get feature maps
		if feature_maps is None:
			feature_maps, labels, list_not_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)
			labels = labels[list_not_null_nodes]

		# Get Reduced features
		X_embedded = get_reduced_features(feature_maps, method)

		# Plot
		do_plot(X_embedded=X_embedded, df_truth=labels , method=method)

	@staticmethod
	def plot_class_relationships_objective_function(data_mode, dataset_name):

		config = reading_user_input_arguments(dataset_name=dataset_name)

		feature_maps, labels, list_not_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)

		for method in ['UMAP', 'TSNE']:
			Visualize.plot_class_relationships(config=config, method=method, data_mode=data_mode, feature_maps=feature_maps, labels=labels[list_not_null_nodes])


	@staticmethod
	def plot_roc_curves_objective_function(input):

		thresh_technique = input[0]
		approach         = input[1]
		dataset_name     = input[2]

		aim1_1 = AIM1_1_TorchXrayVision.run_full_experiment(approach=approach, dataset_name=dataset_name)
		Visualize.plot_roc_curves(config=aim1_1.config , data=aim1_1.test , thresh_technique=thresh_technique)


	@classmethod
	def loop(cls, experiment='roc_curve', data_mode=DataModes.TEST):

		import itertools
		multiprocessing.set_start_method('spawn', force=True)
		from tqdm.notebook import tqdm

		if experiment == 'roc_curve':
			inputs_list = list( itertools.product(ThreshTechList, ['loss', 'logit'], DatasetList.members()) )
			with multiprocessing.Pool(processes=len(inputs_list)) as pool:
				pool.map(cls.plot_roc_curves_objective_function, inputs_list)

		elif experiment == 'class_relationship':
			for dataset_name in tqdm(DatasetList.members()):
				# Also plot the merged datasets figure
				cls.plot_class_relationships_objective_function(dataset_name=dataset_name, data_mode=data_mode)

	def plot_metrics_all_thresh_techniques(self, save_figure=False):

		import matplotlib.pyplot as plt
		import seaborn as sns

		def save_plot():
			save_path = self.config.local_path.joinpath(f'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/')
			save_path.mkdir(parents=True, exist_ok=True)
			for format in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(f'metrics_AUC_ACC_F1.{format}'), format=format, dpi=300)

		def get_metrics():
			columns = pd.MultiIndex.from_product( [['ACC', 'AUC', 'F1'], ['baseline', 'loss', 'logit']])
			metric_df = {}
			for thresh_technique in ['DEFAULT', 'ROC', 'PRECISION_RECALL']:
				output = AIM1_1_TorchXrayVision.get_all_metrics(datasets_list=['CheX', 'NIH', 'PC'], data_mode=DataModes.TEST, thresh_technique=thresh_technique)
				metric_df[thresh_technique] = pd.DataFrame(columns=columns)
				for metric in ['ACC', 'AUC', 'F1']:
					df = pd.DataFrame(dict( baseline=output.loss.baseline.auc_acc_f1.T[metric], loss=output.loss.proposed.auc_acc_f1.T[metric], logit=output.logit.proposed.auc_acc_f1.T[metric]))
					metric_df[thresh_technique][metric] = df.T[output.list_nodes_impacted].T

			return metric_df

		def plot():
			metric_df = get_metrics()
			fig, axes = plt.subplots(3, 3, figsize=(21, 21), sharey=True, sharex=True)
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True, rc=None)

			params = dict(legend=False, fontsize=16, kind='barh')
			for i, thresh_technique in enumerate(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
				metric_df[thresh_technique]['ACC'].plot(ax=axes[i,0], xlabel='ACC', ylabel=thresh_technique, **params)
				metric_df[thresh_technique]['AUC'].plot(ax=axes[i,1], xlabel='AUC', ylabel=thresh_technique, **params)
				metric_df[thresh_technique]['F1' ].plot(ax=axes[i,2], xlabel='F1' , ylabel=thresh_technique, **params)

			plt.legend(loc='lower right', fontsize=16)
			plt.tight_layout()

		plot()
		if save_figure: save_plot()

# endregion
