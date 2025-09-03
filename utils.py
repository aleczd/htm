from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
from IPython import display as disp

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader


import os
import random

import shutil
from cleanfid import fid

from sklearn import metrics

import seaborn as sns

import pickle




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #




# Method to (re)create a directory
def recreate_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)




# Save (pickle) the given object
def save_pkl(obj, name, save_dir="./saves"):
    with open(save_dir + "/" + name + ".pkl", "wb") as f: 
        pickle.dump(obj, f)


# Load a given object by path
def load_pkl(path):
    with open(path, "rb") as f: 
        return pickle.load(f)






# Analyses the policy of a given model, instance by instance
#   <model> is the HTM model being analysed
#   <dataloader> is a PyTorch DataLoader object of the custom (test) dataset used
#   <P> is the token population size of interest
#   <mrgs> is the number of merges to perform
#
#   <fig_dpi> allows scaling of the visualisations
def policy_eval(model, dataloader, P, mrgs=-1, fig_dpi=70):
    # Prepare visualisations
    plt.rcParams["figure.dpi"] = fig_dpi


    # Build a test input
    test_input = next(iter(dataloader))[P].to(model.device)
    test_input = test_input[random.randint(0, test_input.size(0) - 1)].unsqueeze(0)
    print("Test Input: \n\n", test_input, test_input.shape)
    test_vis = dataloader.dataset.visualise(test_input)
    plt.imshow(test_vis[0], cmap=test_vis[1])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("~"*100, "\n")

    # Obtain the testing results
    with torch.no_grad():
        test_output = model(test_input, merges=mrgs)[0]
    print("Test Ouput (" + str(mrgs) + " merges): \n\n", test_output, test_output.shape)
    print("\n\nActions: \n", model.actions, model.actions.shape)
    out_vis = dataloader.dataset.visualise(test_output)
    plt.imshow(out_vis[0], cmap=out_vis[1])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("~"*100, "\n")

    # Overlay to see the difference
    true_img = np.expand_dims(test_vis[0], axis=2)
    pred_img = np.expand_dims(out_vis[0], axis=2)
    overlay = np.concatenate((true_img, pred_img, true_img), axis=2)
    plt.imshow(overlay)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("~"*100, "\n")













# Calculates the FID scores of the models given
# NOTE: Models should be of the form:
#   models = [
#       [<HTM Object>, <Model Title>, <Training Set>, <Testing Set>, <Colour>, <Merges>]
#   ]
def calculate_fid(models, batch_macro, true_dir="./true_images/", samp_dir="./sample_images/", delete_dirs=True):
    outputs = []
    for model_data in models:
        recreate_dir(true_dir)
        recreate_dir(samp_dir)
        true_dl = DataLoader(model_data[3], batch_size=batch_macro, shuffle=False, collate_fn=model_data[3].collate_fn)
        counter = 1
        for batch in true_dl:
            for tp in batch.keys():
                for tens_img in batch[tp]:

                    # Save a true image
                    true_img = Image.fromarray(model_data[3].visualise(tens_img)[0])
                    true_img.save(true_dir + str(counter) + ".png")
                    

                    # Generate a sample
                    gen_img = model_data[0].sample()
                    samp_img = Image.fromarray(model_data[3].visualise(gen_img)[0])
                    samp_img.save(samp_dir + str(counter) + ".png")

                    counter += 1


        # Calculate fid scores    
        score = fid.compute_fid(true_dir, samp_dir, mode="clean", num_workers=0)
        outputs.append(f"{model_data[1]}, FID score: {score}")
    
    disp.clear_output()
    for msg in outputs:
        print(msg)

    if delete_dirs:
        shutil.rmtree(true_dir)
        shutil.rmtree(samp_dir)






# Evaluates the classifier using a confusion matrix, ROC and PR curves
# NOTE: Models should be of the form:
#   models = [
#       [<HTM Object>, <Model Title>, <Training Set>, <Testing Set>, <Colour>, <Merges>]
#   ]
def clf_eval(models, batch_macro):
    test_res = {}
    for model_data in models:
        test_res[model_data[1]] = {
            "pos_prob": np.empty(0),
            "pred_lbl": np.empty(0),
            "true_lbl": np.empty(0)
        }
        valid_dl = DataLoader(model_data[3], batch_size=batch_macro, shuffle=True, collate_fn=model_data[3].collate_fn)
        for batch in valid_dl:
            for tp in batch.keys():
                with torch.no_grad():
                    
                    # merge predicitons take priority at decision threshold
                    clf_logits, clf_lbl = model_data[0](batch[tp].to(model_data[0].device))[-3:-1]
                    sftmx = F.softmax(clf_logits.reshape(-1, 2), dim=1)
                    thresh_mask = sftmx < model_data[0].decision_threshold
                    idx = (thresh_mask == 0).nonzero()
                    pred_lbl = torch.zeros(sftmx.size(0), dtype=torch.long).to(model_data[0].device)
                    pred_lbl[idx[:, 0]] = idx[:, 1]

                    test_res[model_data[1]]["pos_prob"] = np.concat((test_res[model_data[1]]["pos_prob"], sftmx[:, 1].cpu().numpy()), axis=0)
                    test_res[model_data[1]]["pred_lbl"] = np.concat((test_res[model_data[1]]["pred_lbl"], pred_lbl.cpu().numpy()), axis=0)
                    test_res[model_data[1]]["true_lbl"] = np.concat((test_res[model_data[1]]["true_lbl"], clf_lbl.flatten().cpu().numpy()), axis=0)


        print("~~~~~~~~~~~~~~~\t ",model_data[1] , "\t~~~~~~~~~~~~~~~~~\n")     
        print(metrics.classification_report(test_res[model_data[1]]["true_lbl"], 
                                            test_res[model_data[1]]["pred_lbl"], 
                                            target_names=["Leaves", "Merged"],
                                            zero_division=0.0, digits=4))
        print("\n\n")

        confM = metrics.confusion_matrix(test_res[model_data[1]]["true_lbl"], 
                                            test_res[model_data[1]]["pred_lbl"]) 
        cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confM, display_labels=["Leaves", "Merged"])
        cm_disp.plot()
        plt.show()
        print("\n\n")

    # Roc Curves
    fig = plt.figure(figsize=(8, 8))
    for model_data in models:
        fpr, tpr, _ = metrics.roc_curve(test_res[model_data[1]]["true_lbl"], test_res[model_data[1]]["pos_prob"])
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_data[1]} (AUC = {auc:.4})", color=model_data[4])
    plt.plot([0, 1], [0, 1], "--", color="slategray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # PR Curves
    fig = plt.figure(figsize=(8, 8))
    fin_bound = 1
    for model_data in models:
        prec, rec, _ = metrics.precision_recall_curve(test_res[model_data[1]]["true_lbl"], test_res[model_data[1]]["pos_prob"])
        auc = metrics.auc(rec, prec)
        plt.plot(rec, prec, label=f"{model_data[1]} (AUC = {auc:.4})", color=model_data[4])
        bound = len(test_res[model_data[1]]["true_lbl"][test_res[model_data[1]]["true_lbl"]==1]) / len(test_res[model_data[1]]["true_lbl"])
        fin_bound = min(bound, fin_bound)
    plt.plot([0, 1], [fin_bound, fin_bound], "--", color="slategray")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.show()





# Overfitting violin plots
# NOTE: Models should be of the form:
#   models = [
#       [<HTM Object>, <Model Title>, <Training Set>, <Testing Set>, <Colour>, <Merges>]
#   ]
def reconstruction_eval(models, batch_macro, inversion=False, log_scale=True, yLim=None):
    
    # Prepare containers
    test_res = {}
    test_res = {
            "Loss (MSE)": [],    # The actual loss values
            "Dataset": [],       # Training or testing 
            "Model": [],         # Model title
   
        }
    for model_data in models:
        
        # Train losses
        train_dl = DataLoader(model_data[2], batch_size=batch_macro, shuffle=True, collate_fn=model_data[2].collate_fn)
        for batch in train_dl:
            for tp in batch.keys():
                with torch.no_grad():
                    true = batch[tp].to(model_data[0].device)
                    pred = model_data[0](true, merges=model_data[5])[0]
                    loss = torch.mean(((pred - true) ** 2), dim=[1,2])
                    test_res["Loss (MSE)"] += loss.tolist()
                    test_res["Dataset"] += ["Training"] * pred.size(0)
                    test_res["Model"] += [model_data[1]] * pred.size(0)
                
        # Testing losses
        valid_dl = DataLoader(model_data[3], batch_size=batch_macro, shuffle=True, collate_fn=model_data[3].collate_fn)
        for batch in valid_dl:
            for tp in batch.keys():
                with torch.no_grad():
                    true = batch[tp].to(model_data[0].device)
                    pred = model_data[0](true, merges=model_data[5])[0]
                    loss = torch.mean(((pred - true) ** 2), dim=[1,2])
                    test_res["Loss (MSE)"] += loss.tolist()
                    test_res["Dataset"] += ["Testing"] * pred.size(0)
                    test_res["Model"] += [model_data[1]] * pred.size(0)
        
    
    plette = {
    "Training": "skyblue",
    "Testing": "lightsalmon" 
    }

    plette2 = {}
    for model in models:
        plette2[model[1]] = model[4]

    plt.figure(figsize=(24, 12))

    if not inversion:
        plt.title("MSE Loss Between Training & Testing Sets")
        sns.violinplot(data=test_res, x="Model", y="Loss (MSE)", hue="Dataset", fill=True, split=True, linewidth=1, alpha=0.7, inner="quart", log_scale=log_scale, palette=plette)
    else:
        plt.title("MSE Loss Between Two Models for Both Training and Testing Sets")
        sns.violinplot(data=test_res, x="Dataset", y="Loss (MSE)", hue="Model", fill=True, split=True, linewidth=1, alpha=0.7, inner="quart", log_scale=log_scale, palette=plette2)

    if yLim != None:
        plt.ylim(*yLim)

    plt.show()
    