import torch
from torch.utils.data import DataLoader as torch_DataLoader
from typing import Union
import itertools
import torchxrayvision as xrv
from taxonomy.fixed_params import DataModes, ThreshTechList, DatasetList, ExperimentSTAGE
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from taxonomy.utils import Hierarchy

class Data:
	data_loader: torch.utils.data.DataLoader = None
	list_datasets = DatasetList
	
	def __init__(self, data_mode: Union[str, DataModes]):
		
		self.data_mode = data_mode.value if isinstance(data_mode, DataModes) else data_mode
			
		self._d_data      : Optional[xrv.datasets.CheX_Dataset] = None
		self.data_loader  : Optional[torch_DataLoader]    = None
		self.Hierarchy_cls: Optional[Hierarchy] = None
		self.ORIGINAL     : Optional[Findings]  = None
		self.NEW          : Optional[Findings]  = None
		self.labels       : Optional[Labels]    = None
		self.list_findings_names: list 	 = []
	
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
	
	def initialize_findings(self, pathologies: list[str], experiment_stage: str):
		setattr(self, experiment_stage, Findings(experiment_stage=experiment_stage, pathologies=pathologies))
		
		self.list_findings_names.append(experiment_stage)

class Labels:
	def __init__(self, d_data, Hierarchy_cls):
		self.d_data = d_data
		self.labels = pd.DataFrame(d_data.labels, columns=d_data.pathologies) if d_data else pd.DataFrame()
		self.nodes  = self._get_nodes(Hierarchy_cls)
		self.totals = pd.DataFrame(self.d_data.totals()) if d_data else pd.DataFrame()
	
	def _get_nodes(self, Hierarchy_cls):
		
		@dataclass
		class Nodes:
			unique           : List[str] = field(default_factory = list)
			parents          : List[str] = field(default_factory = list)
			exist_in_taxonomy: List[str] = field(default_factory = list)
			not_null         : List[str] = field(default_factory = list)
			impacted         : List[str] = field(default_factory = list)
			node_thresh_tuple: List[str] = field(default_factory = list)
		
		nodes = Nodes()
		nodes.unique   = self.totals.index.to_list()
		nodes.parents  = set(Hierarchy.taxonomy_structure.keys())
		
		if Hierarchy_cls:
			nodes.exist_in_taxonomy = Hierarchy_cls.list_nodes_exist_in_taxonomy
		
		if self.labels.size > 0:
			nodes.not_null          = self.labels.columns[self.labels.count() > 0].to_list()
			nodes.impacted          = [x for x in nodes.not_null if x in nodes.exist_in_taxonomy]
			nodes.node_thresh_tuple = list(itertools.product(nodes.impacted                      , ThreshTechList))
			
		return nodes
	
	
class Findings:
	
	list_metrics   = ['AUC'    , 'ACC' , 'F1'   , 'Threshold']
	list_arguments = ['metrics', 'pred', 'logit', 'truth'     , 'loss']
	
	def __init__(self, pathologies: List[str], experiment_stage: Union[str, ExperimentSTAGE]):
		
		
		self.experiment_stage = ExperimentSTAGE[experiment_stage.upper()] if isinstance(experiment_stage, str) else experiment_stage
		
		self.pathologies = pathologies
		
		# Initializing Metrics & Thresholds
		columns = pd.MultiIndex.from_product([ThreshTechList, self.pathologies], names=['thresh_technique', 'pathologies'])
		
		self.metrics = pd.DataFrame(columns=columns, index=Findings.list_metrics)
		
		# Initializing Arguments & Findings
		self.truth = pd.DataFrame(columns=pathologies)
		
		if self.experiment_stage == ExperimentSTAGE.ORIGINAL:
			self.pred     = pd.DataFrame(columns=pathologies)
			self.logit    = pd.DataFrame(columns=pathologies)
			self.loss     = pd.DataFrame(columns=pathologies)
			self._results = { key : getattr( self, key ) for key in Findings.list_arguments }
		
		elif self.experiment_stage == ExperimentSTAGE.NEW:
			self.pred              = pd.DataFrame( columns=columns )
			self.logit             = pd.DataFrame( columns=columns )
			self.loss              = pd.DataFrame( columns=columns )
			self.hierarchy_penalty = pd.DataFrame( columns=columns )
			self._results = { key : getattr( self, key ) for key in Findings.list_arguments + ['hierarchy_penalty'] }
	
	@property
	def results(self):
		return self._results
	
	@results.setter
	def results(self, value):
		self._results = value
		
		if value is not None:
			for key in Findings.list_arguments:
				setattr(self, key, value[key])
			
			if self.experiment_stage == ExperimentSTAGE.NEW:
				setattr(self, 'hierarchy_penalty', value['hierarchy_penalty'])
