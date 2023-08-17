import argparse
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torchxrayvision as xrv
from taxonomy.utilities.params import ModelNames, DataModes
from taxonomy.utilities.utils import LoadChestXrayDatasets

USE_CUDA = torch.cuda.is_available()


class LoadModelXRV:
	def __init__(self, config):
		self.model    : Optional[torch.nn.Module] = None
		self.modelName: ModelNames                 = config.modelName or ModelNames.ALL_224
		self.chexpert_weights_path: pathlib.Path  = config.path_baseline_CheX_weights
		self.config   : argparse.Namespace        = config

	def load(self, op_threshes: bool=False) -> torch.nn.Module:
		
		def _get_model():
			if self.modelName is ModelNames.BASELINE_JFHEALTHCARE:
				return xrv.baseline_models.jfhealthcare.DenseNet()
			
			elif self.modelName is ModelNames.BASELINE_CHEX:
				return xrv.baseline_models.chexpert.DenseNet(weights_zip=self.chexpert_weights_path)
			
			elif self.modelName is ModelNames.ALL_512:
				return xrv.models.ResNet(weights=self.modelName.value, apply_sigmoid=False)
			
			return xrv.models.DenseNet(weights=self.modelName.value, apply_sigmoid=False)
		
		def _post_process_model():
			
			nonlocal model
			
			if not op_threshes: model.op_threshes = None
			
			if USE_CUDA: model.cuda()
			
		model = _get_model()
		_post_process_model()
		
		return model
	
	@property
	def serve_op_threshes(self):
		return xrv.models.model_urls[self.modelName.value.lower()]['op_threshes']
	
	@property
	def xrv_labels(self):
		return xrv.models.model_urls[self.modelName]['labels']
	
	@property
	def all_models_available(self):
		return ModelNames.members()
	
	@classmethod
	def extract_feature_maps(cls, config: argparse.Namespace, data_mode: DataModes=DataModes.TEST) -> Tuple[np.ndarray, pd.DataFrame, list]:
	
		def _get_model():
			LM = cls(config)
			return LM.load()
		
		def _get_data():
			LD = LoadChestXrayDatasets(config=config, pathologies_in_model=model.pathologies)
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
		
		return feature_maps , pd.DataFrame(truth, columns=model.pathologies), data.labels.nodes.not_null