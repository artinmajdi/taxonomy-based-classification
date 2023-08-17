import multiprocessing


from taxonomy.utilities.utils import TaxonomyXRV, Visualize, CalculateOriginalFindings


# def loop():
#     for dataset_name in DatasetNames.members():
#         for methodName in MethodNames.members():
#             TaxonomyXRV.run_full_experiment(methodName=methodName, dataset_name=dataset_name)

if __name__ == '__main__':

    # TaxonomyXRV.run_full_experiment()

    # TaxonomyXRV.loop_run_full_experiment()

    # Visualize.loop(experiment='class_relationship', data_mode='test')

    # features, truth, _, LD = CalculateOriginalFindings.get_feature_maps(data_mode='test', dataset_name='CheX')

    output = TaxonomyXRV.get_all_metrics_all_thresh_techniques(
        datasets_list=['CheX', 'NIH', 'PC'], data_mode='test')
    
    output.ROC.plot_roc_curves()
