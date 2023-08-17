import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxonomy.utilities.params import DataModes, MethodNames, DatasetNames, EvaluationMetricNames
from taxonomy.utilities.utils import reading_user_input_arguments, TaxonomyXRV, LoadSaveFindings


class Tables:
	
	def __init__(self, jupyter=True, **kwargs):
		self.config = reading_user_input_arguments(jupyter=jupyter, **kwargs)
	
	def get_metrics_per_thresh_techniques(self, save_table=True, data_mode=DataModes.TEST.value,
	                                      thresh_technique='DEFAULT'):
		
		save_path = self.config.local_path.joinpath(
			f'tables/metrics_per_dataset/{thresh_technique}/metrics_{data_mode}.xlsx')
		
		def save(metricsa):
			
			save_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Create a new Excel writer
			with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
				# Write each metric to a different worksheet
				for m in EvaluationMetricNames:
					getattr(metricsa, m.value).to_excel(writer, sheet_name=m.name)
		
		def get():
			columns = pd.MultiIndex.from_product([DatasetNames.members(), MethodNames.members()], names=['dataset', 'methodName'])
			auc = pd.DataFrame(columns = columns)
			f1  = pd.DataFrame(columns = columns)
			acc = pd.DataFrame(columns = columns)
			
			LOGIT    = MethodNames.LOGIT_BASED.name
			LOSS     = MethodNames.LOSS_BASED.name
			BASELINE = MethodNames.BASELINE.name
			
			for dt in DatasetNames.members():
				df = TaxonomyXRV.run_full_experiment(methodName=MethodNames.LOGIT_BASED, dataset_name=dt)
				auc[(dt, LOGIT)]    = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.AUC.name]
				f1[(dt , LOGIT)]    = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.F1.name]
				acc[(dt, LOGIT)]    = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.ACC.name]
				
				auc[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[EvaluationMetricNames.AUC.name]
				f1[(dt , BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[EvaluationMetricNames.F1.name]
				acc[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[EvaluationMetricNames.ACC.name]
				
				df = TaxonomyXRV.run_full_experiment(methodName=MethodNames.LOSS_BASED, dataset_name=dt)
				auc[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.AUC.name]
				f1[(dt , LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.F1.name]
				acc[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.ACC.name]
			
			auc = auc.apply(pd.to_numeric).round(3).replace(np.nan, '')
			f1  = f1.apply(pd.to_numeric).round(3).replace(np.nan , '')
			acc = acc.apply(pd.to_numeric).round(3).replace(np.nan, '')
			
			# region load Data & Model
			@dataclass
			class Metrics:
				auc: pd.DataFrame
				acc: pd.DataFrame
				f1:  pd.DataFrame
			
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
					LD = LoadChestXrayDatasets.get_dataset_unfiltered(update_empty_parent_class=(mode == 'updated'),
					                                                  dataset_name=dname, views=views)
					df2[views] = LD.labels.sum(axis=0).astype(int).replace(0, '')
					
					# Adding the Total row
					df2.loc['Total', views] = LD.labels.shape[0]
				
				return df2.apply(combine_PA_AP, axis=1)
			
			columns = pd.MultiIndex.from_product([[ExperimentSTAGE.ORIGINAL.name, 'updated'], DatasetNames.members()])
			df = pd.DataFrame(columns=columns)
			
			for mode, dname in itertools.product([ExperimentSTAGE.ORIGINAL.name, 'updated'], DatasetNames.members()):
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
	def plot_class_relationships(config: argparse.Namespace, method: str = 'TSNE',
	                             data_mode: DataModes = DataModes.TEST, feature_maps: Optional[np.ndarray] = None,
	                             labels: Optional[Labels] = None) -> None:
		
		path_main = config.local_path.joinpath(f'{config.MLFlow_run_name}/class_relationship')
		
		def get_reduced_features() -> np.ndarray:
			if method.upper() == 'UMAP':
				reducer = umap.UMAP()
				return reducer.fit_transform(feature_maps)
			
			elif method.upper() == 'TSNE':
				from sklearn.manifold import TSNE
				return TSNE(n_components=2, random_state=42).fit_transform(feature_maps)
		
		def do_plot(df_truth: pd.DataFrame) -> None:
			
			def save_plot():
				
				# output path
				filename = f'class_relationship_{method}'
				path = path_main.joinpath(method)
				path.mkdir(parents=True, exist_ok=True)
				
				# Save the plot
				for format in ['png', 'eps', 'svg', 'pdf']:
					plt.savefig(path.joinpath(f'{filename}.{format}'), format=format, dpi=300)
			
			colors = plt.cm.get_cmap('tab20', max(18, len(df_truth.columns)))
			_, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
			axes = axes.flatten()
			
			for i, node in enumerate(df_truth.columns):
				class_indices = df_truth[df_truth[node].eq(1)].index.to_numpy()
				axes[i].scatter(X_embedded[:, 0], X_embedded[:, 1], color='lightgray', alpha=0.2)
				axes[i].scatter(X_embedded[class_indices, 0], X_embedded[class_indices, 1], c=[colors(i)], alpha=0.5,
				                s=20)
				axes[i].set_title(node)
			
			plt.suptitle(f"{method} Visualization for {config.dataset_name} dataset")
			
			# Save the plot
			save_plot()
			
			plt.show()
		
		# Get feature maps
		if feature_maps is None:
			feature_maps, labels, list_not_null_nodes = LoadModelXRV.extract_feature_maps(config=config,
			                                                                              data_mode=data_mode)
			labels = labels[list_not_null_nodes]
		
		# Get Reduced features
		X_embedded = get_reduced_features()
		
		# Plot
		do_plot(df_truth=labels)
	
	@staticmethod
	def plot_class_relationships_objective_function(data_mode, dataset_name):
		
		config = reading_user_input_arguments(dataset_name=dataset_name)
		
		feature_maps, labels, list_not_null_nodes = LoadModelXRV.extract_feature_maps(config=config,
		                                                                              data_mode=data_mode)
		
		for method in ['UMAP', 'TSNE']:
			Visualize.plot_class_relationships(config=config, method=method, data_mode=data_mode,
			                                   feature_maps=feature_maps, labels=labels[list_not_null_nodes])
	
	def plot_metrics_all_thresh_techniques(self, save_figure=False):
		
		import matplotlib.pyplot as plt
		import seaborn as sns
		
		def save_plot():
			save_path = self.config.local_path.joinpath(
				f'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/')
			save_path.mkdir(parents=True, exist_ok=True)
			for format in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(f'metrics_AUC_ACC_F1.{format}'), format=format, dpi=300)
		
		def get_metrics():
			columns = pd.MultiIndex.from_product([EvaluationMetricNames.members(), MethodNames.members()])
			metric_df = {}
			for thresh_technique in ThreshTechList:
				output = TaxonomyXRV.get_all_metrics(datasets_list=DatasetNames.members(),
				                                     data_mode=DataModes.TEST,
				                                     thresh_technique=thresh_technique)
				metric_df[thresh_technique] = pd.DataFrame(columns=columns)
				for metric in EvaluationMetricNames.members():
					df = pd.DataFrame(dict(baseline=output.loss.baseline.auc_acc_f1.T[metric],
					                       loss=output.loss.proposed.auc_acc_f1.T[metric],
					                       logit=output.logit.proposed.auc_acc_f1.T[metric]))
					metric_df[thresh_technique][metric] = df.T[output.list_nodes_impacted].T
			
			return metric_df
		
		def plot():
			metric_df = get_metrics()
			fig, axes = plt.subplots(3, 3, figsize=(21, 21), sharey=True, sharex=True)
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True,
			              rc=None)
			
			params = dict(legend=False, fontsize=16, kind='barh')
			for i, thresh_technique in enumerate(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
				metric_df[thresh_technique][EvaluationMetricNames.ACC.name].plot(ax=axes[i, 0], xlabel=EvaluationMetricNames.ACC.name, ylabel=thresh_technique, **params)
				metric_df[thresh_technique][EvaluationMetricNames.AUC.name].plot(ax=axes[i, 1], xlabel=EvaluationMetricNames.AUC.name, ylabel=thresh_technique, **params)
				metric_df[thresh_technique][EvaluationMetricNames.F1.name].plot(ax=axes[i, 2], xlabel=EvaluationMetricNames.F1.name, ylabel=thresh_technique, **params)
			
			plt.legend(loc='lower right', fontsize=16)
			plt.tight_layout()
		
		plot()
		if save_figure: save_plot()
