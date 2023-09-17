from __future__ import annotations

import itertools
import json
import pathlib
import pickle
from dataclasses import dataclass, field, InitVar
from functools import cached_property, singledispatchmethod, wraps
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as torch_DataLoader

from taxonomy.utilities.params import DatasetNames, ThreshTechList
from taxonomy.utilities.settings import DatasetInfo, Settings

USE_CUDA = torch.cuda.is_available()


@dataclass
class TaxonomyInfo:
	labels: InitVar[pd.DataFrame]
	config: InitVar[Settings]
	graph            : nx.DiGraph = field(init = False)
	classes          : set = field(init = False)
	NON_NULL         : set = field(init = False)
	IMPACTED         : set = field(init = False)
	default_taxonomy : dict = field( init = False )

	def __post_init__(self, labels: pd.DataFrame, config: Settings):

		self.default_taxonomy = { key: set( values ) for key, values in config.dataset.default_taxonomy.items() }
		self.classes = set(labels.columns.to_list())
		self._update(labels)

	@property
	def taxonomy(self):
		return {
			parent: self.default_taxonomy[parent].intersection( self.classes )
			for parent in self.default_taxonomy
			if parent in self.classes
		}

	@cached_property
	def exist_in_taxonomy(self) -> set:

		if len( self.taxonomy ) ==0:
			return set()

		# Adding the parent classes
		eit = set( self.taxonomy.keys() )

		# Adding the children classes
		for value in self.taxonomy.values():
			eit.update(value)

		return eit

	@cached_property
	def node_thresh_tuple(self):
		return list(itertools.product(self.IMPACTED, ThreshTechList.members()))

	def _update(self, labels: pd.DataFrame = None):

		def _construct_graph() -> nx.DiGraph:
			graph = nx.DiGraph( self.taxonomy )
			graph.add_nodes_from( self.classes )
			return graph

		self.graph = _construct_graph()

		if labels and labels.size > 0:
			self.NON_NULL = set( labels.columns[labels.count() > 0].to_list() )

		self.IMPACTED = self.NON_NULL.intersection( self.exist_in_taxonomy )

	def get_children_of(self, parent: str) -> set[str]:
		return self.taxonomy.get( parent, set() )

	def get_parent_of(self, child: str) -> str | None:
		return next( (parent for parent in self.taxonomy if child in self.taxonomy[parent]), None )


@dataclass
class Node:
	name            : str
	taxonomy_info   : TaxonomyInfo
	non_null_indices: dict[str, pd.Series] = None

	def __str__(self):
		return self.name

	@property
	def impacted(self):
		return self.name in self.taxonomy_info.IMPACTED

	@property
	def children(self):
		return self.taxonomy_info.get_children_of(self.name)

	@property
	def parent(self):
		return self.taxonomy_info.get_parent_of(self.name)

	@property
	def is_not_null(self):
		return self.name in self.taxonomy_info.NON_NULL

	@property
	def is_in_taxonomy(self):
		return self.name in self.taxonomy_info.exist_in_taxonomy


@dataclass
class Data:
	config     : Settings
	dataset    : xrv.datasets.Dataset
	labels     : pd.DataFrame     = None
	data_loader: torch_DataLoader = None

	@cached_property
	def taxonomy_info(self) -> TaxonomyInfo:
		return TaxonomyInfo(labels=self.labels, config=self.config)

	@property
	def nodes(self) -> list[Node]:
		return [Node( name = n,
					  taxonomy_info    = self.taxonomy_info,
					  non_null_indices = ~self.labels[n].isna())
					  for n in self.taxonomy_info.classes]

	def update_labels(self, labels: Union[pd.DataFrame, np.ndarray, None] = None) -> 'Data':

		if labels is None:
			self.labels =  pd.DataFrame(self.dataset.labels, columns=self.dataset.pathologies)
		elif isinstance(labels, pd.DataFrame):
			self.labels = labels
		elif isinstance(labels, np.ndarray):
			self.labels = pd.DataFrame(labels, columns=self.dataset.pathologies)
		else:
			raise ValueError("labels must be of type pd.DataFrame, np.ndarray or None")

		return self

	def subset_dataset_using_labels_indices(self) -> 'Data':
		self.dataset = xrv.datasets.SubsetDataset(self.dataset, idxs=self.labels.index)
		return self


@dataclass
class DataTrainTest:
	train: Data = field(default=None)
	test : Data = field(default=None)

@dataclass
class LoadChestXrayDatasets:
	config     : Settings
	datasetInfo: Union[DatasetInfo, None] = None
	dataset    : xrv.datasets.Dataset = field(init = False)
	labels     : pd.DataFrame = field(init = False)
	data       : Data = field(init = False)
	train      : Data = field(init = False)
	test       : Data = field(init = False)

	def load_raw_database(self) -> 'LoadChestXrayDatasets':
		"""
			# RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
				Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
				Annotations of Possible Pneumonia.	Shih, George, Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
				More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
				Challenge site:	https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
				JPG files stored here:	https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9

			# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031
				Dataset website here: https://stanfordmlgroup.github.io/competitions/chexpert/

			# NIH ChestX-ray8 dataset_full. https://arxiv.org/abs/1705.02315
				Dataset release website:
				https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

				Download full size images here:
				https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

				Download resized (224x224) images here:
				https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0

			# PadChest: A large chest threshold_technique-ray image dataset_full with multi-label annotated reports. https://arxiv.org/abs/1901.07441
				Note that images with null labels (as opposed to normal), and images that cannot
				be properly loaded (listed as 'missing' in the code) are excluded, which makes
				the total number of available images slightly less than the total number of image
				files.

				PadChest: A large chest threshold_technique-ray image dataset_full with multi-label annotated reports.
				Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá.
				arXiv preprint, 2019. https://arxiv.org/abs/1901.07441
				Dataset website: https://bimcv.cipf.es/bimcv-projects/padchest/
				Download full size images here: https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850
				Download resized (224x224) images here (recropped):	https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797

			# VinDr-CXR: An open dataset_full of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
				VinBrain Dataset. Nguyen et al., VinDr-CXR: An open dataset_full of chest X-rays with radiologist's annotations
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
				A relabeling of a subset of images from the NIH dataset_full.  The data tables should
				be applied against an NIH download.  A test and validation split are provided in the
				original.  They are combined here, but one or the other can be used by providing
				the original csv to the csvpath argument.

				Chest Radiograph Interpretation with Deep Learning Models: Assessment with
				Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
				Anna Majkowska. Radiology 2020	https://pubs.rsna.org/doi/10.1148/radiol.2019191293
				NIH data can be downloaded here:	https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
		"""

		transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size = 224, engine = "cv2")])

		self.datasetInfo.params_config.update( {'transform': transform} )

		dataset_getter = {
				DatasetNames.NIH       : xrv.datasets.NIH_Dataset,
				DatasetNames.PC        : xrv.datasets.PC_Dataset,
				DatasetNames.CHEXPERT  : xrv.datasets.CheX_Dataset,
				DatasetNames.MIMIC     : xrv.datasets.MIMIC_Dataset,
				DatasetNames.Openi     : xrv.datasets.Openi_Dataset,
				DatasetNames.VinBrain  : xrv.datasets.VinBrain_Dataset,
				DatasetNames.RSNA      : xrv.datasets.RSNA_Pneumonia_Dataset,
				DatasetNames.NIH_Google: xrv.datasets.NIH_Google_Dataset
				}
		self.dataset = dataset_getter[self.datasetInfo.datasetName](**self.datasetInfo.params_config)

		return self

	def _relabel_raw_database(self) -> 'LoadChestXrayDatasets':
		from taxonomy.utilities.model import LoadModelXRV

		# Adding the PatientID if it doesn't exist
		# if "patientid" not in self.dataset.csv:
		# 	self.dataset.csv["patientid"] = [ f"{self.dataset.__class__.__name__}-{i}"
		# 	for i in range(len(self.dataset)) ]
		# Filtering the data.dataset
		# self.dataset.csv = self.dataset.csv[self.dataset.csv['Frontal/Lateral'] == 'Frontal'].reset_index(drop=False)

		# Aligning labels to have the same order as the pathologies' argument.
		xrv.datasets.relabel_dataset( pathologies=LoadModelXRV.model_classes(self.config), dataset=self.dataset, silent=self.config.training.silent )

		return self

	def _create_data_from_dataset(self) -> 'LoadChestXrayDatasets':
		self.data = Data(config=self.config, dataset=self.dataset).update_labels()
		return self

	def _post_process(self) -> 'LoadChestXrayDatasets':

		def update_empty_parent_class_based_on_its_children_classes() -> Optional[None]:

			# Updating the empty parent labels if at least one child label exist
			if self.datasetInfo.datasetName not in [DatasetNames.PC , DatasetNames.NIH]:
				return None

			for parent, children in self.data.taxonomy_info.taxonomy.items():

				if self.data.labels[parent].value_counts().values.shape[0] != 0:
					continue

				print(f"Parent class: {parent} is not labeled. Replacing its true values according to its children presence.")

				# Initializing the parent label to 0
				self.data.labels[parent] = 0

				# If at-least one of the children has a label of 1, then the parent label is 1
				self.data.labels[parent][ self.data.labels[children].sum(axis=1) > 0 ] = 1

		def selecting_non_null_samples():

			for node in self.data.nodes:

				# Skips if node is not a parent node
				if not node.children:
					continue

				# Extracting the samples with a non-null value for the parent truth label
				self.data.labels = self.data.labels[ ~self.data.labels[node].isna() ]
				self.data.subset_dataset_using_labels_indices()

				# Extracting the samples, where for each parent at least one child has a non-null truth label
				self.data.labels = self.data.labels[(~self.data.labels[node.children]).sum( axis=1 ) > 0]
				self.data.subset_dataset_using_labels_indices()

		# Updating the empty parent labels with the child labels
		update_empty_parent_class_based_on_its_children_classes()

		# Selecting non-null samples for impacted pathologies
		selecting_non_null_samples()

		return self

	def train_test_split(self, data: Data) -> tuple[Data, Data]:

		# Splitting the data.dataset into train and test
		idx_train     = data.labels.sample(frac = self.config.dataset.train_test_ratio).index
		dataset_train = xrv.datasets.SubsetDataset( data.dataset, idxs=idx_train )
		train: Data = Data(config=self.config, dataset=dataset_train, labels=data.labels.iloc[idx_train])

		idx_test     = data.labels.drop(idx_train).index
		dataset_test = xrv.datasets.SubsetDataset( data.dataset, idxs=idx_test )
		test: Data = Data(config=self.config, dataset=dataset_test, labels=data.labels.iloc[idx_test])

		return train, test

	@classmethod
	def load_one_dataset(cls, config: Settings, datasetInfo: DatasetInfo) -> Data:

		DT = cls(config=config, datasetInfo=datasetInfo)
		DT.load_raw_database()
		DT._relabel_raw_database()
		DT._create_data_from_dataset()
		DT._post_process()

		return DT.data

	# TODO: need to change all instances where I change something in the labels(dataframe) to change in the dataset.labels (ndarray)
	@classmethod
	def load(cls, config: Settings) -> DataTrainTest:

		dataset_list = [cls(config=config, datasetInfo=di).load_raw_database().dataset for di in config.dataset.datasetInfoList]

		DT = cls(config=config, datasetInfo=None)
		DT.dataset = xrv.datasets.Merge_Dataset( datasets = dataset_list )
		DT._relabel_raw_database()
		DT._create_data_from_dataset()
		DT._post_process()

		# Splitting the data.dataset into train and test
		train, test = DT.train_test_split( DT.data )

		# Creating the data_loader
		train.data_loader = LoadChestXrayDatasets.create_dataloader( config, train.dataset)
		test.data_loader  = LoadChestXrayDatasets.create_dataloader( config, test.dataset )

		return DataTrainTest( train=train, test=test )

	@staticmethod
	def create_dataloader(config: Settings, dataset: xrv.datasets.Dataset):
		data_loader_args = { key: getattr( config, key ) for key in ['batch_size', 'shuffle', 'num_workers'] }
		return torch.utils.data.DataLoader( dataset, pin_memory=USE_CUDA, **data_loader_args )

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


def check_file_exist(func):
	@wraps(func)
	def wrapper(self, file_path: Union[str, pathlib.Path], *args, **kwargs):
		file_path = pathlib.Path(file_path)
		if not file_path.is_file():
			raise FileNotFoundError(f'file_path {file_path} does not exist')
		return func(self, file_path, *args, **kwargs)
	return wrapper


class LoadSaveFile:

	def __init__(self, file_path: Union[str, pathlib.Path]):

		self.file_path = pathlib.Path(file_path)

		if self.file_path.suffix:
			self.file_path.parent.mkdir(parents=True, exist_ok=True)
		else:
			self.file_path.mkdir(parents=True, exist_ok=True)

	def load(self, **kwargs) -> Union[dict, pd.DataFrame, plt.Figure]:

		if self.file_path.suffix == '.pkl':
			with open(self.file_path, 'rb') as f:
				return pickle.load(f)

		if self.file_path.suffix == '.json':
			with open(self.file_path, 'r') as f:
				return json.load(f)

		if self.file_path.suffix == '.csv':
			return pd.read_csv(self.file_path, **kwargs)

		if self.file_path.suffix == '.xlsx':
			return pd.read_excel(self.file_path, **kwargs)

		if self.file_path.suffix == '.png':
			return plt.imread(self.file_path)

		if self.file_path.suffix == '.tif':
			return plt.imread(self.file_path)

		raise NotImplementedError(f'file_type {self.file_path.suffix} is not supported')

	@singledispatchmethod
	def save(self, data: Any, **kwargs):
		raise NotImplementedError(f'file_type {type(data)} is not supported')

	@save.register(dict)
	def _(self, data: dict, **kwargs) -> None:

		assert self.file_path.suffix in {'.pkl', '.json'}, ValueError( f'file type {self.file_path.suffix} is not supported' )

		if self.file_path.suffix == '.pkl':
			with open(self.file_path, 'wb') as f:
				pickle.dump(data, f)

		elif self.file_path.suffix == '.json':
			with open(self.file_path, 'w') as f:
				json.dump(data, f)

		elif self.file_path.suffix == '.xlsx':
			# Recursively call 'save' but with a DataFrame object
			self.save(pd.DataFrame.from_dict(data), self.file_path, **kwargs)

	@save.register(pd.Series)
	def _(self, data: pd.Series, **kwargs):
		assert self.file_path.suffix == '.csv', ValueError(f'file type {self.file_path.suffix} is not supported')
		data.to_csv(self.file_path, **kwargs)

	@save.register(pd.DataFrame)
	def _(self, data: pd.DataFrame, **kwargs) -> None:
		assert self.file_path.suffix == '.xlsx', ValueError(f'file type {self.file_path.suffix} is not supported')
		data.to_excel(self.file_path, **kwargs)

	@save.register(plt.Figure)
	def _(self, data: plt.Figure, file_format: Union[str, list[str]] = None, **kwargs):
		file_format = file_format or [self.file_path.suffix]
		file_format = [file_format] if isinstance(file_format, str) else file_format

		for fmt in (fmt if fmt.startswith('.') else f'.{fmt}' for fmt in file_format):
			data.savefig(self.file_path.with_suffix(fmt), format=fmt.lstrip('.'), dpi=300, **kwargs)


def main():
	config2 = Settings()
	_, Test = LoadChestXrayDatasets.load( config2 )

	print('temp')


if __name__ == '__main__':
	main()
