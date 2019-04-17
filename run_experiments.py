# coding=utf-8
from utils.script_functions import set_parameter
from os import system
from framework import main

if __name__ == "__main__":
    parameters_filepath = "parameters.ini"

    # Keep as all 5
    train_methods = ['natural', 'rfgsm_k', 'dfgsm_k', 'bga_k', 'bca_k', 'grosse']
    evasion_methods = ['natural', 'rfgsm_k', 'dfgsm_k', 'bga_k', 'bca_k', 'grosse']
    experiment_suffix = "demo" # any name you like in case your run multiple experiments

    # this loop will run adversarial training  based on the methods in `train_methods`
    # it will produce len(train_methods) models storing them at `./helper_files`
    for train_method in train_methods:
        set_parameter(parameters_filepath, "general", "load_model_weights", "False")
        set_parameter(parameters_filepath, "general", "experiment_suffix", experiment_suffix)
        set_parameter(parameters_filepath, "general", "training_method", train_method)
        set_parameter(parameters_filepath, "general", "evasion_method", train_method)
        set_parameter(parameters_filepath, "challenge", "eval", "False")
        set_parameter(parameters_filepath, "challenge", "defend", "True")
        set_parameter(parameters_filepath, "challenge", "attack", "False")
        main()

    # This loop will fetch the models trained in the above loop
    # and attack them with the attacks specified in `evasion_methods`
    for train_method in train_methods:
        model_filepath = "./helper_files/[training:{train_meth}|evasion:{train_meth}]_{experiment_suffix}-model.pt".format(
            train_meth=train_method, experiment_suffix=experiment_suffix)

        set_parameter(parameters_filepath, "general", "experiment_suffix", experiment_suffix)
        set_parameter(parameters_filepath, "general", "training_method", train_method)
        set_parameter(parameters_filepath, "general", "load_model_weights", "True")
        set_parameter(parameters_filepath, "general", "model_weights_path", model_filepath)
        set_parameter(parameters_filepath, "challenge", "eval", "True")
        set_parameter(parameters_filepath, "challenge", "defend", "False")
        set_parameter(parameters_filepath, "challenge", "attack", "True")

        for evasion_method in evasion_methods:
            set_parameter(parameters_filepath, "general", "evasion_method", evasion_method)
            main()
