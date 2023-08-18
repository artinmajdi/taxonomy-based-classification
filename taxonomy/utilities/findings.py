import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxonomy.utilities.params import DataModes, MethodNames, DatasetNames, EvaluationMetricNames, ThreshTechList
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
			columns = pd.MultiIndex.from_product([DatasetNames.members(), MethodNames.members()],
			                                     names=['dataset', 'methodName'])
			AUC = pd.DataFrame(columns = columns)
			F1  = pd.DataFrame(columns = columns)
			ACC = pd.DataFrame(columns = columns)
			
			LOGIT    = MethodNames.LOGIT_BASED.name
			LOSS     = MethodNames.LOSS_BASED.name
			BASELINE = MethodNames.BASELINE.name
			
			for dt in DatasetNames.members():
				df = TaxonomyXRV.run_full_experiment(methodName=MethodNames.LOGIT_BASED, dataset_name=dt)
				AUC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]
				
				AUC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]
				
				df = TaxonomyXRV.run_full_experiment(methodName=MethodNames.LOSS_BASED, dataset_name=dt)
				AUC[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[EvaluationMetricNames.F1.name]
				ACC[(dt, LOSS)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]
			
			AUC = AUC.apply(pd.to_numeric).round(3).replace(np.nan, '')
			F1  = F1.apply( pd.to_numeric).round(3).replace(np.nan, '')
			ACC = ACC.apply(pd.to_numeric).round(3).replace(np.nan, '')
			
			# region load Data & Model
			@dataclass
			class Metrics:
				AUC: pd.DataFrame
				ACC: pd.DataFrame
				F1: pd.DataFrame
			
			return Metrics(AUC=AUC, F1=F1, ACC=ACC)
		
		metrics = get()
		
		if save_table:
			save(metrics)
		
		return metrics
	
	def get_table_datasets_samples(self, save_table=True):
		
		save_path = pathlib.Path('tables/metrics_all_datasets/table_datasets_samples.csv')
		
		def get_PA_AP(mode: str, dname: str) -> pd.Series:
			nonlocal mode, dname
			
			def combine_PA_AP(row):
				return '' if (not row.PA) and (not row.AP) else f"{row.PA}/{row.AP}"
			
			df2 = pd.DataFrame(columns=['PA', 'AP'])
			for views in ['PA', 'AP']:
				
				# Getting the dataset for a specific view
				LD = LoadChestXrayDatasets.get_dataset_unfiltered(update_empty_parent_class=(mode == 'updated'), dataset_name=dname, views=views)
				df2[views] = LD.labels.sum(axis=0).astype(int).replace(0, '')
				
				# Adding the Total row
				df2.loc['Total', views] = LD.labels.shape[0]
			
			return df2.apply(combine_PA_AP, axis=1)
	
		cln_list = [ExperimentSTAGE.members(), DatasetNames.members()]
		columns = pd.MultiIndex.from_product(cln_list, names=['mode', 'dataset'])
		df = pd.DataFrame(columns=columns)
		
		for mode, dname in itertools.product(*cln_list):
			df[(mode, dname)] = get_PA_AP(mode=mode, dname=dname)
			
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
				for ft in ['png', 'eps', 'svg', 'pdf']:
					plt.savefig(path.joinpath(f'{filename}.{ft}'), format=ft, dpi=300)
			
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
			save_path = self.config.local_path / 'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/'
			save_path.mkdir(parents=True, exist_ok=True)
			for ft in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(f'metrics_AUC_ACC_F1.{ft}'), format=ft, dpi=300)
		
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
			fig, axes = plt.subplots(3, 3, figsize=(21, 21), sharey=True, sharex=True)  # type: ignore
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True,
			              rc=None)
			
			params = dict(legend=False, fontsize=16, kind='barh')
			for i, thresh_technique in enumerate(['DEFAULT', 'ROC', 'PRECISION_RECALL']):
				metric_df[thresh_technique][EvaluationMetricNames.ACC.name].plot(ax=axes[i, 0],
				                                                                 xlabel=EvaluationMetricNames.ACC.name,
				                                                                 ylabel=thresh_technique, **params)
				metric_df[thresh_technique][EvaluationMetricNames.AUC.name].plot(ax=axes[i, 1],
				                                                                 xlabel=EvaluationMetricNames.AUC.name,
				                                                                 ylabel=thresh_technique, **params)
				metric_df[thresh_technique][EvaluationMetricNames.F1.name].plot(ax=axes[i, 2],
				                                                                xlabel=EvaluationMetricNames.F1.name,
				                                                                ylabel=thresh_technique, **params)
			
			plt.legend(loc='lower right', fontsize=16)
			plt.tight_layout()
		
		plot()
		if save_figure: save_plot()


	@staticmethod
	def plot_metrics(config: argparse.Namespace, metrics: pd.DataFrame, thresh_technique: ThreshTechList , save_figure=True, figsize=(21, 7), font_scale=1.8, fontsize=20):

		def save_plot():
			save_path = self.config.local_path.joinpath(f'figures/auc_acc_f1_all_datasets/{thresh_technique}/')
			save_path.mkdir(parents=True, exist_ok=True)
			for ft in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig( save_path.joinpath( f'metrics_AUC_ACC_F1.{ft}' ), format=ft, dpi=300 )

		def barplot():
			fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)  # type: ignore
			sns.set_theme(style="darkgrid", palette='deep', font='sans-serif', font_scale=1.5, color_codes=True, rc=None)

			params = dict(legend=False, fontsize=16, kind='barh')

			metrics[EvaluationMetricNames.ACC.name].plot(ax = axes[0], title = EvaluationMetricNames.ACC.name, **params)
			metrics[EvaluationMetricNames.AUC.name].plot(ax = axes[1], title = EvaluationMetricNames.AUC.name, **params)
			metrics[EvaluationMetricNames.F1.name].plot(ax  = axes[2], title = EvaluationMetricNames.F1.name , **params)
			plt.legend(loc='upper right', fontsize=16)
			plt.tight_layout()


		def heatmap():
			import seaborn as sns
			import matplotlib.pyplot as plt

			sns.set(font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper', color_codes=True, rc=None)

			fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)  # type: ignore
			params = dict(annot=True, fmt=".3f", linewidths=.5, cmap='YlGnBu', cbar=False, annot_kws={"size": fontsize})

			for i, m in enumerate(EvaluationMetricNames.members()):
				sns.heatmap(data=metrics[m],ax=axes[i], **params)
				axes[i].set_title(m, fontsize=int(1.5*fontsize), fontweight='bold')
				axes[i].tick_params(axis='both', which='major', labelsize=fontsize)

			plt.tight_layout()

		heatmap()

		if save_figure:
			save_plot()