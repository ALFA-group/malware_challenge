# coding=utf-8
##
# This script is to be used by organizers for evaluating submissions.
##
from utils.script_functions import set_parameter, create_tex_tables
from shutil import copyfile

from os import system
import glob
import os
import re
from framework import main


# this dir contains the submitted aes.npy files, each is suffixed with the method name after _
# e.g., ./attack_submissions/aes_methoda.npy
#       ./attack_submissions/aes_methodb.npy
#
ATTACK_DIR='./attack_submissions'

# this dir will contains the submitted model files (*.pt) each is suffixed with the method name after by _
# e.g., ./defend_submissions/[training:methoda|.pt
#       ./defend_submissions/[training:methodb|.pt
#
DEFEND_DIR='./defend_submissions'


def eval_attack_submssions():
    parameters_filepath = "./helper_files/parameters_attack.ini"
    model_filepath = "./helper_files/[training:natural|evasion:dfgsm_k]_demo-model.pt"
    for attack_filepath in glob.glob(os.path.join(ATTACK_DIR, '*.npy') ):
        if attack_filepath.split('/')[-1] == "aes.npy": continue
        submission_name = attack_filepath.split('/')[-1].split('_')[-1][:-4]
        print("Evaluating {}'s submission".format(submission_name))
        copyfile(attack_filepath, os.path.join(ATTACK_DIR, 'aes.npy'))
        set_parameter(parameters_filepath, "general", "experiment_suffix", submission_name)
        set_parameter(parameters_filepath, "general", "training_method", "secret_model")
        set_parameter(parameters_filepath, "general", "load_model_weights", "True")
        set_parameter(parameters_filepath, "general", "model_weights_path", model_filepath)
        set_parameter(parameters_filepath, "challenge", "eval", "True")
        set_parameter(parameters_filepath, "challenge", "defend", "False")
        set_parameter(parameters_filepath, "challenge", "attack", "True")
        set_parameter(parameters_filepath, "general", "evasion_method", submission_name)
        set_parameter(parameters_filepath, "challenge", "adv_examples_path", ATTACK_DIR)
        main(config_file=parameters_filepath)

    create_tex_tables(filespath="./result_files")
    print("Check evasion latex table file in results_files")


def eval_defend_submissions():
    parameters_filepath = "./helper_files/parameters_holdout.ini"
    evasion_methods = ['natural', 'rfgsm_k', 'dfgsm_k', 'bga_k', 'bca_k', 'grosse']

    for model_filepath in glob.glob(os.path.join(DEFEND_DIR, '*.pt') ):
        submission_name = re.search("\[training:.*\|", model_filepath).group(0)[:-1].split(':')[-1]
        print("Evaluating {}'s submission".format(submission_name))
        set_parameter(parameters_filepath, "general", "experiment_suffix", submission_name)
        set_parameter(parameters_filepath, "general", "training_method", submission_name)
        set_parameter(parameters_filepath, "general", "load_model_weights", "True")
        set_parameter(parameters_filepath, "general", "model_weights_path", model_filepath)
        set_parameter(parameters_filepath, "challenge", "eval", "True")
        set_parameter(parameters_filepath, "challenge", "defend", "True")
        set_parameter(parameters_filepath, "challenge", "attack", "False")
        for evasion_method in evasion_methods:
            set_parameter(parameters_filepath, "general", "evasion_method", evasion_method)
            main(config_file=parameters_filepath)

    create_tex_tables(filespath="./result_files")



if __name__ == "__main__":
    #eval_defend_submissions()
    eval_attack_submssions()
