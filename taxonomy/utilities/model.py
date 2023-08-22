import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torchxrayvision as xrv
from taxonomy.utilities.params import DataModes, ModelWeightNames


USE_CUDA = torch.cuda.is_available()


def extract_feature_maps(config: argparse.Namespace, data_mode: DataModes=DataModes.TEST) -> Tuple[np.ndarray, pd.DataFrame, list]:

	def _get_model():
		LM = LoadModelXRV(config)
		return LM.load()

	def _get_data():
		from taxonomy.utilities.data import LoadChestXrayDatasets
		LD = LoadChestXrayDatasets( config=config )
		LD.load()
		return getattr(LD, data_mode.value)

	def process_one_batch(batch_data):

		# Getting the data and its corresponding true labels
		device = 'cuda' if USE_CUDA else 'cpu'
		images = batch_data["img" ].to(device)

		# Get feature maps
		feats = model.features2(images) if hasattr(model, "features2") else model.features(images)
		return feats.reshape(len(feats), -1).detach().cpu(), batch_data["lab" ].to(device).detach().cpu()

	def looping_over_all_batches(data_loader, n_batches_to_process) -> Tuple[np.ndarray, np.ndarray]:

		d_features, d_truth  = [], []
		for batch_idx, batch_data in enumerate(tqdm(data_loader)):
			if n_batches_to_process and (batch_idx >= n_batches_to_process):
				break
			features_batch, truth_batch = process_one_batch(batch_data)
			d_features.append(features_batch)
			d_truth.append(truth_batch)

		return np.concatenate(d_features), np.concatenate(d_truth)

	model = _get_model()
	data  = _get_data()

	with torch.no_grad():  # inference_mode no_grad
		feature_maps, truth = looping_over_all_batches(data_loader=data.data_loader, n_batches_to_process=config.n_batches_to_process)

	return feature_maps , pd.DataFrame(truth, columns=model.pathologies), data.labels.nodes.non_null


@dataclass
class LoadModelXRV:
	config               : argparse.Namespace
	model                : torch.nn.Module = field(default_factory = lambda: None)
	chexpert_weights_path: pathlib.Path    = field(default_factory = lambda: None)
	modelName            : ModelWeightNames      = field(default_factory = lambda: ModelWeightNames.ALL_224)

	def __post_init__(self):
		self.modelName            : ModelWeightNames   = getattr(self.config, 'modelName', ModelWeightNames.ALL_224)
		self.chexpert_weights_path: pathlib.Path = getattr(self.config, 'PATH_CHEXPERT_WEIGHTS', None)

	def load(self, op_threshes: bool=False) -> torch.nn.Module:
		
		def _get_model():

			if 'baseline' in self.modelName.value:

				if self.modelName is ModelWeightNames.BASELINE_JFHEALTHCARE:
					return xrv.baseline_models.jfhealthcare.DenseNet()

				elif self.modelName is ModelWeightNames.BASELINE_CHEX:
					return xrv.baseline_models.chexpert.DenseNet(weights_zip=self.chexpert_weights_path)

			else:

				if 'resnet' in self.modelName.value:
					return xrv.models.ResNet(weights=self.modelName.value, apply_sigmoid=False)

				else:
					return xrv.models.DenseNet(weights=self.modelName.value, apply_sigmoid=False)
		
		model = _get_model()

		if not op_threshes:
			model.op_threshes = None

		if USE_CUDA:
			model.cuda()
		
		return model
	
	@property
	def serve_op_threshes(self):
		return xrv.models.model_urls[self.modelName.value.lower()]['op_threshes']
	
	@staticmethod
	def model_classes(config: argparse.Namespace):
		return xrv.models.model_urls[config.modelName]['labels']
