from main.aims.taxonomy.utils_taxonomy import *

output = AIM1_1_TorchXrayVision.get_all_metrics_all_thresh_techniques(
    datasets_list=['CheX', 'NIH', 'PC'], data_mode='test')

output.ROC.plot_roc_curves()
