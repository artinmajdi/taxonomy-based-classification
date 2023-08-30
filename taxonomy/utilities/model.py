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
from taxonomy.utilities.settings import Settings

USE_CUDA = torch.cuda.is_available()

@dataclass
class LossFunctionOptions:
	binary_crossentropy: torch.nn.BCELoss(reduction='none')
	bce                 : torch.nn.BCELoss(reduction='none')
	bce_with_logits     : torch.nn.BCEWithLogitsLoss(reduction='none')


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
	config: Settings
	model : torch.nn.Module = None

	def load(self, op_threshes: bool=False):

		modelName = self.config.model.modelName

		def _get_model() -> torch.nn.Module:
			if modelName == ModelWeightNames.BASELINE_JFHEALTHCARE:
				return xrv.baseline_models.jfhealthcare.DenseNet()

			elif modelName == ModelWeightNames.BASELINE_CHEX:
				return xrv.baseline_models.chexpert.DenseNet(weights_zip=self.config.model.chexpert_weights_path)

			elif modelName.value.startswith('resnet'):
				return xrv.models.ResNet(weights=modelName.full_name, apply_sigmoid=False)

			elif modelName.value.startswith('densenet'):
				return xrv.models.DenseNet(weights=modelName.full_name, apply_sigmoid=False)

			raise NotImplementedError(f"Model {modelName} not implemented")

		self.model = _get_model()

		self.model.op_threshes = xrv.models.model_urls[modelName.value]['op_threshes'] if op_threshes else None

		if USE_CUDA:
			self.model.cuda()

		return self
	
	@staticmethod
	def model_classes(config: argparse.Namespace):
		return xrv.models.model_urls[config.modelName]['labels']
