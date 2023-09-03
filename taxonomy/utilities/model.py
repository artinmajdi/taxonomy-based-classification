from dataclasses import dataclass
from typing import Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torchxrayvision as xrv
from taxonomy.utilities.data import Data, DataTrainTest
from taxonomy.utilities.params import DataModes, ModelWeightNames
from taxonomy.utilities.settings import Settings

USE_CUDA = torch.cuda.is_available()

@dataclass
class LossFunctionOptions:
	binary_crossentropy: torch.nn.BCELoss(reduction='none')
	bce                : torch.nn.BCELoss(reduction='none')
	bce_with_logits    : torch.nn.BCEWithLogitsLoss(reduction='none')


ModelType: TypeAlias = Union[xrv.models.DenseNet, xrv.models.ResNet]

@dataclass
class LoadModelXRV:
	config: Settings
	model : ModelType = None

	def load(self, op_threshes: bool=False) -> 'LoadModelXRV':

		modelName = self.config.model.modelName

		def _get_model() -> ModelType:
			if modelName == ModelWeightNames.BASELINE_JFHEALTHCARE:
				model: xrv.models.DenseNet = xrv.baseline_models.jfhealthcare.DenseNet()

			elif modelName == ModelWeightNames.BASELINE_CHEX:
				model: xrv.models.DenseNet = xrv.baseline_models.chexpert.DenseNet(weights_zip=self.config.model.chexpert_weights_path)

			elif modelName.value.startswith('resnet'):
				model: xrv.models.ResNet = xrv.models.ResNet(weights=modelName.full_name, apply_sigmoid=False)

			elif modelName.value.startswith('densenet'):
				model: xrv.models.DenseNet = xrv.models.DenseNet(weights=modelName.full_name, apply_sigmoid=False)

			else:
				raise NotImplementedError(f"Model {modelName} not implemented")

			return model

		self.model = _get_model()
		self.model.op_threshes = xrv.models.model_urls[modelName.value]['op_threshes'] if op_threshes else None

		if USE_CUDA:
			self.model.cuda()

		return self

	@staticmethod
	def model_classes(config: Settings) -> list:
		return xrv.models.model_urls[config.modelName]['labels']

	@classmethod
	def extract_feature_maps(cls, config: Settings, data_mode: DataModes) -> Tuple[np.ndarray, pd.DataFrame, list]:

		def _get_data() -> Data:
			from taxonomy.utilities.data import LoadChestXrayDatasets
			LD: DataTrainTest = LoadChestXrayDatasets.load( config=config )
			return LD.train if data_mode is DataModes.TRAIN else LD.test

		model: ModelType = cls(config = config).load().model
		data: Data = _get_data()

		def process_one_batch(batch_data) -> Tuple[np.ndarray, np.ndarray]:

			# Getting the data and its corresponding true labels
			device = 'cuda' if USE_CUDA else 'cpu'
			images = batch_data["img" ].to(device)

			# Get feature maps
			feats = model.features2(images) if hasattr(model, "features2") else model.features(images)
			features_batch: np.ndarray = feats.reshape(len(feats), -1).detach().cpu()
			truth_batch   : np.ndarray = batch_data["lab" ].to(device).detach().cpu()
			return features_batch, truth_batch

		def looping_over_all_batches() -> Tuple[np.ndarray, np.ndarray]:

			batches_to_process = config.training.batches_to_process
			d_features, d_truth  = [], []
			for batch_idx, batch_data in enumerate(tqdm(data.data_loader)):

				if batches_to_process and (batch_idx >= batches_to_process):
					break

				features_batch, truth_batch = process_one_batch(batch_data)
				d_features.append(features_batch)
				d_truth.append(truth_batch)

			return np.concatenate(d_features), np.concatenate(d_truth)

		with torch.no_grad():  # inference_mode no_grad
			feature_maps, truth = looping_over_all_batches()

		return feature_maps , pd.DataFrame(truth, columns=model.pathologies), data.labels.nodes.non_null
