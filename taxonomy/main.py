import multiprocessing


from taxonomy.utilities.utils import TaxonomyXRV, Visualize, CalculateOriginalFindings


# def loop():
#     for datasetName in DatasetNames.members():
#         for methodName in TechniqueNames.members():
#             TaxonomyXRV.run_full_experiment(methodName=methodName, datasetName=datasetName)

if __name__ == '__main__':

    # TaxonomyXRV.run_full_experiment()

    # TaxonomyXRV.loop_run_full_experiment()

    # Visualize.loop(experiment='class_relationship', data_mode='test')

    # features, truth, _, LD = CalculateOriginalFindings.get_feature_maps(data_mode='test', datasetName='CheX')

    output = TaxonomyXRV.get_all_metrics_all_thresh_techniques(
        datasets_list=['CheX', 'NIH', 'PC'], data_mode='test')
    
    output.ROC.plot_roc_curves()
