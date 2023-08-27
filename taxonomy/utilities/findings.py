import argparse
import itertools
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import umap
from matplotlib import pyplot as plt

from taxonomy.utilities.data import Findings, Labels, LoadChestXrayDatasets, Metrics, LoadSaveFile
from taxonomy.utilities.model import extract_feature_maps
from taxonomy.utilities.params import DataModes, DatasetNames, EvaluationMetricNames, ExperimentStageNames, \
	TechniqueNames, ThreshTechList
from taxonomy.utilities.settings import get_settings
from taxonomy.utilities.utils import TaxonomyXRV

def calculating_threshold_and_metrics(findings: Findings) -> Metrics:

	classes = findings.data.nodeData.nodes.classes
	metrics = Metrics(pathologies=classes)

	def calculating_threshold_and_metrics_per_node(node: str, findings: Findings, thresh_technique: ThreshTechList) -> Findings:

		truth = findings.modelOutputs.truth_values
		pred = findings.modelOutputs.pred_values

		def calculating_optimal_thresholds(y, yhat):

			if thresh_technique == ThreshTechList.DEFAULT:
				metrics.THRESHOLD = 0.5

			if thresh_technique == ThreshTechList.ROC:
				fpr, tpr, th = sklearn.metrics.roc_curve(y, yhat)
				metrics.THRESHOLD = th[np.argmax( tpr - fpr )]

			if thresh_technique == ThreshTechList.PRECISION_RECALL:
				ppv, recall, th = sklearn.metrics.precision_recall_curve(y, yhat)
				f_score = 2 * (ppv * recall) / (ppv + recall)
				metrics.THRESHOLD = th[np.argmax( f_score )]

		def calculating_metrics(y, yhat, x):
			metrics.AUC = sklearn.metrics.roc_auc_score(y, yhat)
			metrics.ACC = sklearn.metrics.accuracy_score(y, yhat >= findings.metrics[x, node]['Threshold'])
			metrics.F1  = sklearn.metrics.f1_score      (y, yhat >= findings.metrics[x, node]['Threshold'])

		# Finding the indices where the truth is not nan
		non_null = ~np.isnan(truth[node])
		truth_notnull = truth[node][non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
			pred = findings.modelOutputs.pred_values[node]
			pred_notnull = pred[non_null].to_numpy()

			calculating_optimal_thresholds( y = truth_notnull, yhat = pred_notnull)
			calculating_metrics(y = truth_notnull, yhat = pred_notnull , x = thresh_technique)

		return findings

	for x in ThreshTechList:
		for node in DATA.pathologies:
			DATA = calculating_threshold_and_metrics_per_node(node=node, findings=DATA, thresh_technique=x)

	return DATA


class Tables:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	def get_metrics_per_thresh_techniques(self, save_table=True, data_mode=DataModes.TEST.value, thresh_technique='DEFAULT'):

		save_path = self.config.PATH_LOCAL.joinpath(
			f'tables/metrics_per_dataset/{thresh_technique}/metrics_{data_mode}.xlsx')

		def save(metricsa):

			save_path.parent.mkdir(parents=True, exist_ok=True)

			# Create a new Excel writer
			with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
				# Write each metric to a different worksheet
				for m in EvaluationMetricNames:
					getattr(metricsa, m.value).to_excel(writer, sheet_name=m.name)

		def get():
			columns = pd.MultiIndex.from_product([DatasetNames.members(), TechniqueNames.members()],
			                                     names=['dataset_full', 'methodName'])
			AUC = pd.DataFrame(columns = columns)
			F1  = pd.DataFrame(columns = columns)
			ACC = pd.DataFrame(columns = columns)

			LOGIT    = TechniqueNames.LOGIT_BASED.name
			LOSS     = TechniqueNames.LOSS_BASED.name
			BASELINE = TechniqueNames.BASELINE.name

			for dt in DatasetNames.members():
				df = TaxonomyXRV.run_full_experiment(methodName=TechniqueNames.LOGIT_BASED, datasetName=dt)
				AUC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]

				AUC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]

				df = TaxonomyXRV.run_full_experiment(methodName=TechniqueNames.LOSS_BASED, datasetName=dt)
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

	@staticmethod
	def get_dataset_unfiltered(**kwargs):
		config = get_settings(**kwargs)
		LD = LoadChestXrayDatasets( config=config )
		LD.load_raw_database()
		LD.relabel_raw_database()
		LD.update_empty_parent_class_based_on_its_children_classes()
		return LD

	def get_table_datasets_samples(self, save_table=True):

		save_path = pathlib.Path('tables/metrics_all_datasets/table_datasets_samples.csv')

		def get_PA_AP(mode: str, dname: str) -> pd.Series:
			# nonlocal mode, dname

			def combine_PA_AP(row):
				return '' if (not row.PA) and (not row.AP) else f"{row.PA}/{row.AP}"

			df2 = pd.DataFrame(columns=['PA', 'AP'])
			for views in ['PA', 'AP']:

				# Getting the dataset_full for a specific view
				LD = Tables.get_dataset_unfiltered( datasetName=dname, views=views )
				df2[views] = LD.dataset_full.labels.sum( axis=0 ).astype( int ).replace( 0, '' )

				# Adding the Total row
				df2.loc['Total', views] = LD.dataset_full.labels.shape[0]

			return df2.apply(combine_PA_AP, axis=1)

		cln_list = [ExperimentStageNames.members(), DatasetNames.members()]
		columns = pd.MultiIndex.from_product(cln_list, names=['mode', 'dataset_full'])
		df = pd.DataFrame(columns=columns)

		for mode, dname in itertools.product(*cln_list):
			df[(mode, dname)] = get_PA_AP(mode=mode, dname=dname)

		if save_table:
			LoadSaveFindings(self.config, save_path).save(df)

		return df


class Visualize:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	@staticmethod
	def plot_class_relationships(config: argparse.Namespace, method: str = 'TSNE',
	                             data_mode: DataModes = DataModes.TEST, feature_maps: Optional[np.ndarray] = None,
	                             labels: Optional[Labels] = None) -> None:

		path_main = config.PATH_LOCAL.joinpath(f'{config.DEFAULT_FINDING_FOLDER_NAME}/class_relationship')

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

			plt.suptitle(f"{method} Visualization for {config.datasetName} dataset_full")

			# Save the plot
			save_plot()

			plt.show()

		# Get feature maps
		if feature_maps is None:
			feature_maps, labels, list_non_null_nodes = extract_feature_maps(config=config,
			                                                                              data_mode=data_mode)
			labels = labels[list_non_null_nodes]

		# Get Reduced features
		X_embedded = get_reduced_features()

		# Plot
		do_plot(df_truth=labels)

	@staticmethod
	def plot_class_relationships_objective_function(data_mode, datasetName):

		config = get_settings(datasetName=datasetName)

		feature_maps, labels, list_non_null_nodes = extract_feature_maps(config=config, data_mode=data_mode)

		for method in ['UMAP', 'TSNE']:
			Visualize.plot_class_relationships(config=config, method=method, data_mode=data_mode,
			                                   feature_maps=feature_maps, labels=labels[list_non_null_nodes])

	def plot_metrics_all_thresh_techniques(self, save_figure=False):

		def save_plot():
			save_path = self.config.PATH_LOCAL / 'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/'
			save_path.mkdir(parents=True, exist_ok=True)
			for ft in ['png', 'eps', 'svg', 'pdf']:
				plt.savefig(save_path.joinpath(f'metrics_AUC_ACC_F1.{ft}'), format=ft, dpi=300)

		def get_metrics():
			columns = pd.MultiIndex.from_product([EvaluationMetricNames.members(), TechniqueNames.members()])
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
			save_path = config.PATH_LOCAL.joinpath(f'figures/auc_acc_f1_all_datasets/{thresh_technique}/')
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