# coding=utf-8
"""
Python module for performing adversarial training for malware detection
"""
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.utils import load_parameters, stack_tensors
from utils.script_functions import merge_metrics
from datasets.datasets import load_data
from inner_maximizers.inner_maximizers import inner_maximizer
from nets.ff_classifier import build_ff_classifier
from blindspot_coverage.covering_number import CoveringNumber
import time
import json
import numpy as np


def main(config_file="parameters.ini"):
    # Step 1. Load configuration
    parameters = load_parameters(config_file)
    is_cuda = eval(parameters["general"]["is_cuda"])
    if is_cuda:
        # gotcha On some platforms, modifying os.environ will not modify the system environment
        os.environ["CUDA_VISIBLE_DEVICES"] = parameters["general"]["gpu_device"]

    assertion_message = "Set this flag off to train models."
    assert eval(parameters['dataset']['generate_feature_vector_files']) is False, assertion_message

    log_interval = int(parameters["general"]["log_interval"])
    num_epochs = int(parameters["hyperparam"]["ff_num_epochs"])
    is_synthetic_dataset = eval(parameters["general"]["is_synthetic_dataset"])

    training_method = parameters["general"]["training_method"]
    evasion_method = parameters["general"]["evasion_method"]
    experiment_suffix = parameters["general"]["experiment_suffix"]
    experiment_name = "[training:%s|evasion:%s]_%s" % (training_method, evasion_method,
                                                       experiment_suffix)

    adv_example_filepath = parameters["challenge"]["adv_examples_path"]

    print("Training Method:%s, Evasion Method:%s" % (training_method, evasion_method))

    seed_val = int(parameters["general"]["seed"])

    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    evasion_iterations = int(parameters['hyperparam']['evasion_iterations'])

    save_every_epoch = eval(parameters['general']['save_every_epoch'])

    #train_model_from_scratch = eval(parameters['general']['train_model_from_scratch'])
    load_model_weights = eval(parameters['general']['load_model_weights'])
    model_weights_path = parameters['general']['model_weights_path']

    # Step 2. Load training and test data
    train_dataloader_dict, valid_dataloader_dict, test_dataloader_dict, num_features = load_data(
        parameters)

    # set the bscn metric
    num_samples = len(train_dataloader_dict["malicious"].dataset)
    bscn = CoveringNumber(num_samples, num_epochs * num_samples,
                          train_dataloader_dict["malicious"].batch_size)

    if load_model_weights:
        print("Loading Model Weights From: {path}".format(path=model_weights_path))
        model = torch.load(model_weights_path)

    else:
        # Step 3. Construct neural net (N) - this can be replaced with any model of interest
        model = build_ff_classifier(
            input_size=num_features,
            hidden_1_size=int(parameters["hyperparam"]["ff_h1"]),
            hidden_2_size=int(parameters["hyperparam"]["ff_h2"]),
            hidden_3_size=int(parameters["hyperparam"]["ff_h3"]))
    # gpu related setups
    if is_cuda:
        torch.cuda.manual_seed(int(parameters["general"]["seed"]))
        model = model.cuda()

    # Step 4. Define loss function and optimizer  for training (back propagation block in Fig 2.)
    loss_fct = nn.NLLLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=float(parameters["hyperparam"]["ff_learning_rate"]))

    def train(epoch):
        model.train()
        total_correct = 0.
        total_loss = 0.
        total = 0.

        current_time = time.time()

        if is_synthetic_dataset:
            # since generation of synthetic data set is random, we'd like them to be the same over epochs
            torch.manual_seed(seed_val)
            random.seed(seed_val)

        for batch_idx, ((bon_x, bon_y), (mal_x, mal_y)) in enumerate(
                zip(train_dataloader_dict["benign"], train_dataloader_dict["malicious"])):
            # Check for adversarial learning
            mal_x = inner_maximizer(
                mal_x, mal_y, model, loss_fct, iterations=evasion_iterations, method=training_method)

            # stack input
            if is_cuda:
                x = Variable(stack_tensors(bon_x, mal_x).cuda())
                y = Variable(stack_tensors(bon_y, mal_y).cuda())
            else:
                x = Variable(stack_tensors(bon_x, mal_x))
                y = Variable(stack_tensors(bon_y, mal_y))

            # forward pass
            y_model = model(x)

            # backward pass
            optimizer.zero_grad()
            loss = loss_fct(y_model, y).mean()
            loss.backward()
            optimizer.step()

            # predict pass
            _, predicted = torch.topk(y_model, k=1)
            correct = predicted.data.eq(y.data.view_as(predicted.data)).cpu().sum()

            # metrics
            total_loss += loss.data[0] * len(y)
            total_correct += correct
            total += len(y)

            bscn.update_numerator_batch(batch_idx, mal_x)

            if batch_idx % log_interval == 0:
                print("Time Taken:", time.time() - current_time)
                current_time = time.time()
                print(
                    "Train Epoch ({}) | Batch ({}) | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tBatch Accuracy: {:.1f}%\t BSCN: {:.12f}".
                        format(epoch, batch_idx, batch_idx * len(x),
                               len(train_dataloader_dict["malicious"].dataset) +
                               len(train_dataloader_dict["benign"].dataset),
                               100. * batch_idx / len(train_dataloader_dict["benign"]), loss.data[0],
                               100. * correct / len(y), bscn.ratio()))

        model_filename = "{name}_epoch_{e}".format(name=experiment_name, e=epoch)

        if save_every_epoch:
            torch.save(model, os.path.join("model_weights", model_filename))

    def check_one_category(category="benign", dset_type='test', is_evade=False,
                           evade_method='dfgsm_k'):
        """
        test the model in terms of loss and accuracy on category, this function also allows to perform perturbation
        with respect to loss to evade
        :param category: benign or malicious dataset
        :param dset_type: 'val', 'test', or 'train' dataset
        :param is_evade: to perform evasion or not
        :param evade_method: evasion method (we can use on of the inner maximier methods), it is only relevant if is_evade
          is True
        :return:
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        evasion_mode = ""

        if is_synthetic_dataset:
            # since generation of synthetic data set is random, we'd like them to be the same over epochs
            torch.manual_seed(seed_val)
            random.seed(seed_val)

        if dset_type == 'train':
            dataloader = train_dataloader_dict[category]
        elif dset_type == 'val':
            dataloader = valid_dataloader_dict[category]
        elif dset_type == 'test':
            dataloader = test_dataloader_dict[category]
        else:
            raise Exception("Invalid Dataset type")

        for batch_idx, (x, y) in enumerate(dataloader):
            #
            if is_evade:
                x = inner_maximizer(
                    x, y, model, loss_fct, iterations=evasion_iterations, method=evade_method)
                evasion_mode = "(evasion using %s)" % evade_method
            # stack input
            if is_cuda:
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)

            # forward pass
            y_model = model(x)

            # loss pass
            loss = loss_fct(y_model, y).mean()

            # predict pass
            _, predicted = torch.topk(y_model, k=1)
            correct = predicted.data.eq(y.data.view_as(predicted.data)).cpu().sum()

            # metrics
            total_loss += loss.data[0] * len(y)
            total_correct += correct
            total += len(y)

        print("{} set for {} {}: Average Loss: {:.4f}, Accuracy: {:.2f}%".format(
            dset_type, category, evasion_mode, total_loss / total,
                                               total_correct * 100. / total))

        return total_loss, total_correct, total

    def test(epoch, dset_type='test'):
        """
        Function to be used for both testing and validation
        :param epoch: current epoch
        :param dset_type: 'train', 'test' , or 'val'
        :return: average total loss, dictionary of the metrics for both bon and mal samples
        """
        # test for accuracy and loss
        bon_total_loss, bon_total_correct, bon_total = check_one_category(
            category="benign", is_evade=False, dset_type=dset_type)
        mal_total_loss, mal_total_correct, mal_total = check_one_category(
            category="malicious", is_evade=False, dset_type=dset_type)

        # test for evasion on malicious sample
        evade_mal_total_loss, evade_mal_total_correct, evade_mal_total = check_one_category(
            category="malicious", is_evade=True, evade_method=evasion_method, dset_type=dset_type)

        total_loss = bon_total_loss + mal_total_loss
        total_correct = bon_total_correct + mal_total_correct
        total = bon_total + mal_total

        print("{} set overall: Average Loss: {:.4f}, Accuracy: {:.2f}%".format(
            dset_type, total_loss / total, total_correct * 100. / total))

        metrics = {
            "bscn_num_pts": bscn.num_pts(),
            "bscn_exp_pts": bscn.exp_num_pts(),
            "mal": {
                "total_loss": mal_total_loss,
                "total_correct": mal_total_correct,
                "total": mal_total,
                "evasion": {
                    "total_loss": evade_mal_total_loss,
                    "total_correct": evade_mal_total_correct,
                    "total": evade_mal_total
                }
            },
            "bon": {
                "total_loss": bon_total_loss,
                "total_correct": bon_total_correct,
                "total": bon_total
            }
        }
        print(metrics)

        return (bon_total_loss + max(mal_total_loss, evade_mal_total_loss)) / total, metrics

    def process_adv_examples(evade_method='dfgsm_k', mode='gen'):
        """
        This function is used for the `attack` track challenge for two purposes
        With mode='gen', it is meant to craft transferable adversarial examples and store them to a numpy array
        With mode='eval', it loads up the examples from the numpy array and evaluates them on the tested model
        Note, ADV Examples are only crafted for malicious files
        :param evade_method: evasion method (participants can implement their own), here we use `dfgsm_k` as an example
        :param mode: 'gen' to generate and store the adv examples or 'eval' to load them and evaluate
        :return:
        """
        model.eval()
        category = "malicious"
        total_loss = 0
        total_correct = 0
        total = 0

        adv_file = os.path.join(adv_example_filepath, 'aes.npy')
        xs_adv = [] if mode == 'gen' else np.load(adv_file)
        # to be inline with the code base, the attack dataset will also be
        # decomposed into train, val, and test. However, all these subsets will be
        # used as part of the attack challenge.
        xs_adv_offset = 0
        for dset_type in ['train', 'val', 'test']:
            if dset_type == 'train':
                dataloader = train_dataloader_dict[category]
            elif dset_type == 'val':
                dataloader = valid_dataloader_dict[category]
            elif dset_type == 'test':
                dataloader = test_dataloader_dict[category]

            # to impose the order of the crafted examples, we manually loop over the dataset
            # instead of using the dataloader' sampler
            batch_size = dataloader.batch_size
            num_pts = len(dataloader.dataset)
            num_batches = (num_pts + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                # get the batch data
                bstart = batch_idx * batch_size
                bend = min(num_pts, bstart + batch_size)
                x, y = zip(*[dataloader.dataset[i] for i in range(bstart, bend)])
                x = torch.stack(x, dim=0)
                y = torch.LongTensor(y)

                if mode == 'gen':
                    # put your method here
                    # ---------------------------
                    x_adv = inner_maximizer(
                            x, y, model, loss_fct, iterations=evasion_iterations, method=evade_method)
                    # ---------------------------

                else:
                    x_adv = torch.from_numpy(
                        xs_adv[xs_adv_offset + batch_idx * batch_size:
                               xs_adv_offset + (batch_idx + 1) * batch_size, :])

                # stack input
                if is_cuda:
                    x_adv = Variable(x_adv.cuda())
                    y = Variable(y.cuda())
                else:
                    x_adv = Variable(x_adv)
                    y = Variable(y)

                # forward pass
                y_model = model(x_adv)

                # loss pass
                loss = loss_fct(y_model, y).mean()

                # predict pass
                _, predicted = torch.topk(y_model, k=1)
                correct = predicted.data.eq(y.data.view_as(predicted.data)).cpu().sum()

                # metrics
                total_loss += loss.data[0] * len(y)
                total_correct += correct
                total += len(y)

                # let's save the adversarial examples
                _x = x.numpy()
                _x_adv = x_adv.cpu().data.numpy() if is_cuda else x_adv.data.numpy()
                assert np.allclose(np.logical_and(_x, _x_adv), _x), "perturbation constraint violated"
                if mode == 'gen':
                    xs_adv = xs_adv + [_x_adv]

            xs_adv_offset += num_pts

        if mode == 'gen':
            np.save(adv_file, np.concatenate(xs_adv, axis=0))

        # we keep the same structure of metrics for compatibility
        metrics = {
            "bscn_num_pts": 1,
            "bscn_exp_pts": 1,
            "mal": {
                "total_loss": 1,
                "total_correct": 1,
                "total": 1,
                "evasion": {
                    "total_loss": total_loss,
                    "total_correct": total_correct,
                    "total": total
                }
            },
            "bon": {
                "total_loss": 1,
                "total_correct": 1,
                "total": 1
            }
        }

        return metrics

    if not os.path.exists("result_files"):
        os.mkdir("result_files")
    _metrics = None

    # Starter kit for Defend Challenge
    if not eval(parameters["challenge"]["eval"]) and eval(parameters["challenge"]["defend"]):
        best_valid_loss = float("inf")
        for _epoch in range(num_epochs):
            # train
            train(_epoch)
            # validate
            valid_loss, _ = test(_epoch, dset_type='val')
            # keep the best parameters w.r.t validation and check the test set
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                _, _metrics = test(_epoch, dset_type='test')
                bscn_to_save = bscn.ratio()
                with open(os.path.join("result_files", "%s_bscn.txt" % experiment_name), "w") as f:
                    f.write(str(bscn_to_save))
                torch.save(model, os.path.join("helper_files", "%s-model.pt" % experiment_name))
            elif _epoch % log_interval == 0:
                test(_epoch, dset_type='test')

    # starter kit for Attack challenge:
    # participants can modify the code highlighted in the `process_adv_examples` function
    if not eval(parameters["challenge"]["eval"]) and eval(parameters["challenge"]["attack"]):
        _metrics = process_adv_examples(evade_method=evasion_method, mode='gen')

    # Code for submission evaluations (this code will be run by the organizers),
    # we are relasing it here for transparency
    # for evaluating submissions under the Attack track
    if eval(parameters["challenge"]["eval"]) and eval(parameters["challenge"]["attack"]):
        _metrics = process_adv_examples(evade_method=evasion_method, mode='eval')

    # for evaluating submissions under the Defend track
    # For compatibility with the code above, our hold-out dataset will also
    # be splitted into test, validation, and train. This is why we evaluate them all below.
    if eval(parameters["challenge"]["eval"]) and eval(parameters["challenge"]["defend"]):
        # report results on all datasets
        _, _metrics = test(0, dset_type='test')
        _, _metrics_t = test(0, dset_type='train')
        _, _metrics_v = test(0, dset_type='val')
        _metrics = merge_metrics([_metrics_t, _metrics, _metrics_v])

    with open(os.path.join("result_files", experiment_name + ".json"), "w") as result_file:
        json.dump(_metrics, result_file)


if __name__ == "__main__":
    main()
