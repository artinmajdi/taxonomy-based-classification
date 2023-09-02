
from taxonomy.utilities.utils import TaxonomyXRV


output = TaxonomyXRV.get_all_metrics_all_thresh_techniques(
    datasets_list=['CheX', 'NIH', 'PC'], data_mode='test')

output.ROC.plot_roc_curves()
