from dataclasses import dataclass, field
from functools import cached_property, singledispatch, singledispatchmethod, wraps
from typing import Optional, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torchxrayvision as xrv
from matplotlib import pyplot as plt

from taxonomy.utilities.data import Data, LoadSaveFile
from taxonomy.utilities.params import EvaluationMetricNames, ExperimentStageNames, \
	TechniqueNames, ThreshTechList
from taxonomy.utilities.settings import Settings
from taxonomy.utilities.model import ModelType

Array_Series: TypeAlias = Union[np.ndarray, pd.Series]


def get_y_yhat_flag(*args, **kwargs) -> Tuple[Array_Series, Array_Series, bool]:
	""" Extract y and yhat from keyword arguments or positional arguments. """

	# Extract y and yhat from keyword arguments
	y, yhat, RAISE_ValueError_IF_EMPTY = kwargs.get( 'y' ), kwargs.get( 'yhat' ), kwargs.get(
		'RAISE_ValueError_IF_EMPTY', False )

	# If y and yhat are not passed as keyword arguments, attempt to get them from positional arguments.
	if y is None and args:
		y, *args = args

	if yhat is None and args:
		yhat, *args = args

	return y, yhat, RAISE_ValueError_IF_EMPTY


def validating_label_vectors_decorator(func):
	@wraps( func )
	def wrapper(*args, **kwargs):

		y, yhat, _ = get_y_yhat_flag( *args, **kwargs )

		if y is None or yhat is None:
			raise ValueError( "y and yhat must be passed as keyword arguments or positional arguments" )

		if y.shape != yhat.shape:
			raise ValueError( "y and yhat must have the same shape" )

		if y.dtype != yhat.dtype:
			raise ValueError( "y and yhat must be of the same type" )

		if not isinstance( y, (np.ndarray, pd.Series) ):
			raise ValueError( "y and yhat must be numpy arrays or pandas series" )

		if isinstance( y, np.ndarray ) and len( y.shape ) == 2:
			if y.shape[1] != 1:
				raise ValueError( "y & yhat must be 1D arrays" )
			y = y.flatten()
			yhat = yhat.flatten()

		kwargs['y'], kwargs['yhat'] = y, yhat
		return func( *args, **kwargs )

	return wrapper


@validating_label_vectors_decorator
def remove_null_samples(y: Array_Series, yhat: Array_Series, RAISE_ValueError_IF_EMPTY: bool = False) -> Tuple[
	Array_Series, Array_Series]:
	""" Filter out null samples and check if calculation should proceed. """

	non_null = ~np.isnan( y ) if isinstance( y, np.ndarray ) else y.notnull()

	if RAISE_ValueError_IF_EMPTY and (not non_null.any()):
		raise ValueError( "y and yhat must contain at least one non-null value" )

	return y[non_null], yhat[non_null]


@dataclass
class ROC:
	y: np.ndarray
	yhat: np.ndarray
	REMOVE_NULL: bool = True
	THRESHOLD: float = field( default=None )
	FPR: np.ndarray = field( default=None )
	TPR: np.ndarray = field( default=None )

	def __post_init__(self):
		if self.REMOVE_NULL:
			self.y, self.yhat = remove_null_samples( y=self.y, yhat=self.yhat, RAISE_ValueError_IF_EMPTY=False )

	def calculate(self) -> 'ROC':
		if self.y.size > 0:
			fpr, tpr, thr = sklearn.metrics.roc_curve( self.y, self.yhat )
			self.FPR, self.TPR, self.THRESHOLD = fpr, tpr, thr[np.argmax( tpr - fpr )]
		return self


@dataclass
class PrecisionRecall:
	y: np.ndarray
	yhat: np.ndarray
	REMOVE_NULL: bool = True
	THRESHOLD: float = field( default=None )
	PPV: np.ndarray = field( default=None )
	RECALL: np.ndarray = field( default=None )
	F_SCORE: np.ndarray = field( default=None )

	def __post_init__(self):
		if self.REMOVE_NULL:
			self.y, self.yhat = remove_null_samples( y=self.y, yhat=self.yhat, RAISE_ValueError_IF_EMPTY=False )

	def calculate(self) -> 'PrecisionRecall':
		if self.y.size > 0:
			ppv, recall, th = sklearn.metrics.precision_recall_curve( self.y, self.yhat )
			f_score = 2 * (ppv * recall) / (ppv + recall)
			self.THRESHOLD, self.PPV, self.RECALL, self.F_SCORE = th[np.argmax( f_score )], ppv, recall, f_score
		return self


@dataclass
class CalculateMetrics:
	y: np.ndarray
	yhat: np.ndarray
	REMOVE_NULL: bool = True
	thresh_technique: ThreshTechList = ThreshTechList.ROC

	def __post_init__(self):
		if self.REMOVE_NULL:
			self.y, self.yhat = remove_null_samples( y=self.y, yhat=self.yhat, RAISE_ValueError_IF_EMPTY=False )

	@cached_property
	def THRESHOLD(self) -> float:

		if self.thresh_technique == ThreshTechList.DEFAULT:
			return 0.5

		if self.thresh_technique == ThreshTechList.ROC:
			return ROC( y=self.y, yhat=self.yhat, REMOVE_NULL=False ).calculate().THRESHOLD

		if self.thresh_technique == ThreshTechList.PRECISION_RECALL:
			return PrecisionRecall( y=self.y, yhat=self.yhat, REMOVE_NULL=False ).calculate().THRESHOLD

		raise NotImplementedError( f"ThreshTechList {self.thresh_technique} not implemented" )

	@cached_property
	def AUC(self) -> float:
		return sklearn.metrics.roc_auc_score( self.y, self.yhat )

	@cached_property
	def ACC(self) -> float:
		return sklearn.metrics.accuracy_score( self.y, self.yhat > self.THRESHOLD )

	@cached_property
	def F1(self) -> float:
		return sklearn.metrics.f1_score( self.y, self.yhat > self.THRESHOLD )


@dataclass
class ModelOutputs:
	""" Class for storing model-related findings. """
	truth_values: pd.DataFrame = field( default_factory = pd.DataFrame )
	loss_values : pd.DataFrame = field( default_factory = pd.DataFrame )
	logit_values: pd.DataFrame = field( default_factory = pd.DataFrame )
	pred_values : pd.DataFrame = field( default_factory = pd.DataFrame )

	def save(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'ModelOutputs':

		output_path = config.output.path / f'{experiment_stage}/model_outputs.xlsx'
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with pd.ExcelWriter( output_path ) as writer:
			self.truth_values.to_excel( writer, sheet_name = 'truth_values' )
			self.loss_values.to_excel( writer , sheet_name = 'loss_values' )
			self.logit_values.to_excel( writer, sheet_name = 'logit_values' )
			self.pred_values.to_excel( writer , sheet_name = 'pred_values' )
		return self

	def load(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'ModelOutputs':

		input_path = config.output.path / f'{experiment_stage}/model_outputs.xlsx'

		if input_path.is_file():
			self.truth_values = pd.read_excel( input_path, sheet_name  = 'truth_values')
			self.loss_values  = pd.read_excel( input_path, sheet_name  = 'loss_values' )
			self.logit_values = pd.read_excel( input_path, sheet_name  = 'logit_values')
			self.pred_values  = pd.read_excel( input_path, sheet_name  = 'pred_values' )
		return self


Array_Series_DataFrame: TypeAlias = Union[np.ndarray, pd.Series, pd.DataFrame]


@dataclass
class Metrics:
	ACC      : Union[pd.Series, float] = field( default = None )
	AUC      : Union[pd.Series, float] = field( default = None )
	F1       : Union[pd.Series, float] = field( default = None )
	THRESHOLD: Union[pd.Series, float] = field( default = None )

	def save(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'Metrics':

		def get_formatted(metric):
			return metric if isinstance(metric, pd.Series) else pd.Series([metric])

		output_path = config.output.path / f'{experiment_stage}/metrics.xlsx'
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with pd.ExcelWriter( output_path ) as writer:
			get_formatted(self.ACC).to_excel( writer, sheet_name = 'acc' )
			get_formatted(self.AUC).to_excel( writer, sheet_name = 'auc' )
			get_formatted(self.F1).to_excel(  writer, sheet_name = 'f1' )
			get_formatted(self.THRESHOLD).to_excel( writer, sheet_name = 'threshold' )
		return self


	def load(self, config: Settings, experiment_stage: ExperimentStageNames) -> 'Metrics':

		def set_formatted(metric):
			if isinstance(metric, pd.Series) and len(metric) == 1:
				return metric.iloc[0]
			return metric

		input_path = config.output.path / f'{experiment_stage}/metrics.xlsx'

		if input_path.is_file():
			self.ACC = set_formatted(pd.read_excel(input_path, sheet_name = 'acc', index_col = 0, squeeze = True))
			self.AUC = set_formatted(pd.read_excel(input_path, sheet_name = 'auc', index_col = 0, squeeze = True))
			self.F1  = set_formatted(pd.read_excel(input_path, sheet_name = 'f1' , index_col = 0, squeeze = True))
			self.THRESHOLD = set_formatted(pd.read_excel(input_path, sheet_name = 'threshold', index_col = 0, squeeze = True))
		return self


	@classmethod
	def calculate(cls, config: Settings, REMOVE_NULL: bool=True, **kwargs) -> 'Metrics':
		# sourcery skip: raise-specific-error
		try:
			y    = kwargs['model_outputs'].truth_values if 'model_outputs' in kwargs else kwargs['y']
			yhat = kwargs['model_outputs'].pred_values  if 'model_outputs' in kwargs else kwargs['yhat']

		except KeyError as e:
			" The KeyError would occur if the else clauses are reached and kwargs does not contain the keys 'y' or 'yhat'."
			raise KeyError(f"Key {e} not found in kwargs") from e

		except AttributeError as e:
			"The AttributeErrorwould occur if the model_outputs object does not have the attributes truth_values or pred_values."
			raise AttributeError(f"Attribute {e} not found in kwargs") from e

		except TypeError as e:
			" A TypeError would occur if you attempt an operation on a variable that is of an inappropriate type. In this case, if model_outputs is a basic data type like an integer, string, etc., that does not support attribute access."
			raise TypeError(f"TypeError: {e}") from e

		except Exception as e:
			raise Exception(f"Exception: {e}") from e


		return cls._calculate( y=y, yhat=yhat, config=config, REMOVE_NULL=REMOVE_NULL )


	@classmethod
	def _calculate(cls, y: Array_Series_DataFrame, yhat: Array_Series_DataFrame, config: Settings,
		  REMOVE_NULL: bool = True) -> 'Metrics':

		thresh_technique = config.hyperparameter_tuning.thresh_technique

		if isinstance( y, pd.Series ):
			y, yhat = y.to_numpy(), yhat.to_numpy()

		if isinstance( y, np.ndarray ):
			return cls()._calculate_1D_data( y=y, yhat=yhat, REMOVE_NULL=REMOVE_NULL,
											 thresh_technique=thresh_technique )

		elif isinstance( y, pd.DataFrame ):
			metrics = cls()
			metrics.THRESHOLD = pd.Series( index=y.columns )
			metrics.AUC = pd.Series( index=y.columns )
			metrics.ACC = pd.Series( index=y.columns )
			metrics.F1 = pd.Series( index=y.columns )

			for clm in y.columns:
				mts = cls()._calculate_1D_data( y=y[clm], yhat=yhat[clm], REMOVE_NULL=REMOVE_NULL,
												thresh_technique=thresh_technique )

				metrics.THRESHOLD[clm] = mts.THRESHOLD
				metrics.AUC[clm] = mts.AUC
				metrics.ACC[clm] = mts.ACC
				metrics.F1[clm] = mts.F1

			return metrics

		raise NotImplementedError( f"Metrics not implemented for type {type( y )}" )


	def _calculate_1D_data(self, y: np.ndarray, yhat: np.ndarray, REMOVE_NULL: bool = True,
						   thresh_technique: ThreshTechList = ThreshTechList.ROC) -> 'Metrics':

		calculateMetrics = CalculateMetrics( y=y, yhat=yhat, REMOVE_NULL=REMOVE_NULL,
											 thresh_technique=thresh_technique )

		self.THRESHOLD = calculateMetrics.THRESHOLD
		self.AUC       = calculateMetrics.AUC
		self.ACC       = calculateMetrics.ACC
		self.F1        = calculateMetrics.F1
		return self

@dataclass
class Findings:
	""" Class for storing overall findings including configuration, data, and metrics. """
	config          : Settings
	data            : Data                 = field( default = None )
	model           : ModelType            = field( default = None )
	experiment_stage: ExperimentStageNames = ExperimentStageNames.ORIGINAL
	model_outputs   : ModelOutputs         = field( default = None, init = False)
	metrics         : Metrics              = field( default = None, init = False)


@dataclass
class FindingsTrainTest:
	train: Findings = field( default=None )
	test: Findings = field( default=None )


@dataclass
class FindingsAllTechniques:
	loss: FindingsTrainTest = field( default=None )
	logit: FindingsTrainTest = field( default=None )
	baseline: FindingsTrainTest = field( default=None )

	def _get_obj(self):
		for objName in ['loss', 'logit', 'baseline']:
			obj = getattr( self, objName )
			if obj is not None:
				return obj
		raise ValueError(
			'No suitable object has been initialized. Please initialize either "baseline", "loss", or "logit".' )

	@property
	def nodes(self):
		return self._get_obj().test.data.nodes

	@property
	def config(self):
		return self._get_obj().test.config

	def plot_roc_curves(self, save_figure=True, figsize=(15, 15), font_scale=1.8, fontsize=20, labelpad=0):

		impacted_nodes = self.nodes.IMPACTED
		list_parent_nodes = list( self.nodes.taxonomy.keys() )
		truth = self.baseline.test.model_outputs.truth_values

		# Set up the grid
		def setup_plot():
			nonlocal fig, axes, n_rows, n_cols

			# Set a seaborn style for visually appealing plots
			sns.set( font_scale=font_scale, font='sans-serif', palette='colorblind', style='darkgrid', context='paper',
					 color_codes=True, rc=None )

			# Set up the grid
			n_nodes, n_cols = len( impacted_nodes ), 3
			n_rows = int( np.ceil( n_nodes / n_cols ) )

			# Set up the figure and axis
			fig, axes = plt.subplots( n_rows, n_cols, figsize=figsize, sharey=True, sharex=True )  # type: ignore
			axes = axes.flatten()

			return fig, axes, n_rows, n_cols

		fig, axes, n_rows, n_cols = setup_plot()

		def plot_per_node(node, idx):

			row_idx = idx // n_cols
			col_idx = idx % n_cols
			ax = axes[idx]

			# Calculate the ROC curve and AUC
			def get_fpr_tpr_auc(pred_node, truth_node, technique: TechniqueNames, roc_auc):
				nonlocal technique

				def get():
					nonlocal fpr, tpr
					# mask = ~truth_node.isnull()
					mask = ~np.isnan( truth_node )
					truth_notnull = truth_node[mask].to_numpy()

					if (len( truth_notnull ) > 0) and (np.unique( truth_notnull ).size == 2):
						fpr, tpr, _ = sklearn.metrics.roc_curve( truth_notnull, pred_node[mask] )
						return fpr, tpr
					return None, None

				fpr, tpr = get()
				return sns.lineplot( x=fpr, y=tpr, label=f'{technique} AUC = {roc_auc:.2f}', linewidth=2, ax=ax )

			# Plot the ROC curve
			lines, labels = [], []

			for methodName in TechniqueNames.members():
				data = getattr( self, methodName.lower() )
				technique = TechniqueNames[methodName]

				line = get_fpr_tpr_auc( pred_node=data.pred[node], truth_node=truth[node], technique=technique,
										roc_auc=data.auc_acc_f1[node][EvaluationMetricNames.AUC.name] )
				lines.append( line.lines[-1] )
				labels.append( line.get_legend_handles_labels()[1][-1] )

			# Customize the plot
			ax.plot( [0, 1], [0, 1], linestyle='--', linewidth=2 )
			ax.set_xlabel( 'False Positive Rate', fontsize=fontsize,
						   labelpad=labelpad ) if row_idx == n_rows - 1 else ax.set_xticklabels( [] )
			ax.set_ylabel( 'True Positive Rate', fontsize=fontsize,
						   labelpad=labelpad ) if col_idx == 0 else ax.set_yticklabels( [] )
			ax.legend( loc='lower right', fontsize=12 )
			ax.set_xlim( [0.0, 1.0] )
			ax.set_ylim( [0.0, 1.05] )

			leg = ax.legend( lines, labels, loc='lower right', fontsize=fontsize, title=node )
			plt.setp( leg.get_title(), fontsize=fontsize )

			# Set the background color of the plot to gray if the node is a parent node
			if node in list_parent_nodes:
				ax.set_facecolor( 'xkcd:light grey' )

			fig.suptitle( 'ROC Curves', fontsize=int( 1.5 * fontsize ), fontweight='bold' )
			plt.tight_layout()

		def postprocess():
			# Remove any empty plots in the grid
			for empty_idx in range( idx + 1, n_rows * n_cols ):
				axes[empty_idx].axis( 'off' )

			plt.tight_layout()

		# Loop through each disease and plot the ROC curve
		for idx, node in enumerate( self.nodes.IMPACTED ):
			plot_per_node( node, idx )

		# Postprocess the plot
		postprocess()

		# Save the plot
		if save_figure:
			file_path = self.config.output.path / f'figures/roc_curve_all_datasets/{self.config.hyperparameter_tuning.thresh_technique}/roc_curve_all_datasets.png'
			LoadSaveFile( file_path ).save( data=fig )


'''
class Tables:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	def get_metrics_per_thresh_techniques(self, save_table=True, data_mode=DataModes.TEST.value, thresh_technique='DEFAULT'):

		from taxonomy.utilities.utils import TaxonomyXRV

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

			LOGIT    = TechniqueNames.LOGIT.modelName
			LOSS     = TechniqueNames.LOSS.modelName
			BASELINE = TechniqueNames.BASELINE.name

			for dt in DatasetNames.members():
				df = TaxonomyXRV.run_full_experiment(methodName=TechniqueNames.LOGIT, datasetName=dt)
				AUC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, LOGIT)] = getattr(df, data_mode).NEW.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]

				AUC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.AUC.name]
				F1[ (dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.F1.name]
				ACC[(dt, BASELINE)] = getattr(df, data_mode).ORIGINAL.metrics[thresh_technique].loc[ EvaluationMetricNames.ACC.name]

				df = TaxonomyXRV.run_full_experiment(methodName=TechniqueNames.LOSS, datasetName=dt)
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
		return LoadChestXrayDatasets( config=config, datasetInfo=config.dataset.datasetInfoList[0]).get_dataset_unfiltered()

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
			LoadSaveFile(file_path).save(data=df)

		return df


class Visualize:

	def __init__(self, jupyter=True, **kwargs):
		self.config = get_settings(jupyter=jupyter, **kwargs)

	@staticmethod
	def plot_class_relationships(config: Settings, method: str = 'TSNE',
	                             data_mode: DataModes = DataModes.TEST, feature_maps: Optional[np.ndarray] = None,
	                             labels: pd.DataFrame = None) -> None:

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
			feature_maps, labels, list_non_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)
			labels = labels[list_non_null_nodes]

		# Get Reduced features
		X_embedded = get_reduced_features()

		# Plot
		do_plot(df_truth=labels)

	@staticmethod
	def plot_class_relationships_objective_function(data_mode, datasetName):

		config = get_settings(datasetName=datasetName)

		feature_maps, labels, list_non_null_nodes = LoadModelXRV.extract_feature_maps(config=config, data_mode=data_mode)

		for method in ['UMAP', 'TSNE']:
			Visualize.plot_class_relationships(config=config, method=method, data_mode=data_mode,
			                                   feature_maps=feature_maps, labels=labels[list_non_null_nodes])

	def plot_metrics_all_thresh_techniques(self, save_figure=False):

		def get_metrics():
			from taxonomy.utilities.utils import TaxonomyXRV

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

		if save_figure:
			save_path = self.config.output.path / 'final/metrics_all_datasets/fig_metrics_AUC_ACC_F1_all_thresh_techniques/metrics_AUC_ACC_F1.png'
			LoadSaveFile(file_path).save(data=fig, file_format=['png', 'eps', 'svg', 'pdf'])


	@staticmethod
	def plot_metrics(config: Settings, metrics: pd.DataFrame, thresh_technique: ThreshTechList , save_figure=True, figsize=(21, 7), font_scale=1.8, fontsize=20):


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

'''
