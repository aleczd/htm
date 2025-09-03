import math
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as disp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import pickle
import datetime
import os

from sklearn import mixture












# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



# The Hierarchical Token Merger
class HTM(nn.Module):

    ######################################### CORE #########################################


    # Constructor
    def __init__(self,
                 inp_dim,                         # The input (token) dimensions
                 emb_dim,                         # The dimensions used for the embedding space
                 mu_hidn = 256,                   # The hidden dimensions in the (un)merger
                 clf_hidn = 256,                  # The hidden dimensions in the classifier
                 plcy_hidn = 256,                 # The hidden dimensions in the policy network
                 policy_v = "learned",            # Denotes which policy version to use (between 'heuristic' & 'learned')
                                                        # (Can be overriden by the train function)

                 noise_std = 0.01,                # Std. of the gaussian noise applied after lifting/merging  
                 mask_val = -9e20,                # Mask value for the logits matrix (must be a very low negative number) 
                 sftmx_eps = 1e-20,               # Used to avoid zero softmax outputs when computing entropy (must be a very small positive decimal)    
                 std_eps = 1e-20,                 # Used to prevent division by zero when normalising (baselining) the reinforce advantages (must be a very small positive decimal)   
                 mvg_adv_sz = 10_000,             # Size of the tensor tracking the most recent advantages (used in baselining via normalisation)
                 gen_step_limit = 100,            # Max number of steps allowed during generation (ensures termination)
                 decision_threshold = 0.5,        # Decision threshold used during sampling 
                 device = torch.device("cpu")     
                 ):
        
        # Initiate superclass properties
        super(HTM, self).__init__()

        # Base Parameters
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.mu_hidn = mu_hidn
        self.clf_hidn = clf_hidn
        self.plcy_hidn = plcy_hidn
        self.policy_v = policy_v

        self.noise_std = noise_std
        self.mask_val = mask_val
        self.sftmx_eps = sftmx_eps
        self.std_eps = std_eps
        self.mvg_adv_sz = mvg_adv_sz
        self.gen_step_limit = gen_step_limit
        self.decision_threshold = decision_threshold

        self.device = device


        # Forward Pass Memory 
        self.batch_dim = 0                                 # The micro batch dimension (B) of the input tensor
        self.n_inp_tk = 0                                  # The token population dimension (P) of the input tensor
        
        self.logits_mtx = None                             # The logits adjacency matrix used in selecting actions
        self.active_tokens = None                          # The indeces of active tokens that exist in the population
        self.actions = None                                # The action (indeces) of which tokens are merged
        self.step_losses = None                            # The one-step reconstruction losses container
        self.entropies = None                              # The entropy container

        self.log_probs = None                              # Stores the log of the probabilities of the actions occuring under the learned policy
        self.rewards = None                                # The reward for these actions
        self.mvg_adv = torch.empty(0).to(self.device)      # Stores the most recent advantages for baselining

        

        # Training Memory/Utilities
        self.logger_fn = lambda p, v: v if ((p != None) and (p != False)) else None            # Lambda function for visualisation compatibility (ensures plot data are of consistent size)
        self.plot_log = None                                                                   # Dictionary containing the training plot data
        self.hyper_log = None                                                                  # Dictionary containing all of the model's (hyper)parameters
        self.ufi = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                           # Unique File Identifier for saving and loading trained models (can be renamed)
        self.reg = None                                                                        # Flags used for log scaling errors (can be overriden by the training method)
        self.gen = True


        # Networks / Learnable Parameters
        self.lift = nn.Sequential(
            nn.Linear(self.inp_dim, self.emb_dim)
        ).to(self.device)

        self.unlift = nn.Sequential(
            nn.Linear(self.emb_dim, self.inp_dim)
        ).to(self.device)

        self.encoder = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.mu_hidn),
            nn.ReLU(),
            nn.Linear(self.mu_hidn, self.mu_hidn),
            nn.ReLU(),
            nn.Linear(self.mu_hidn, self.emb_dim)
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.mu_hidn),
            nn.ReLU(),
            nn.Linear(self.mu_hidn, self.mu_hidn),
            nn.ReLU(),
            nn.Linear(self.mu_hidn, 2 * self.emb_dim)
        ).to(self.device)

        self.clf = nn.Sequential(
            nn.Linear(self.emb_dim, self.clf_hidn),
            nn.ReLU(),
            nn.Linear(self.clf_hidn, self.clf_hidn),
            nn.ReLU(),
            nn.Linear(self.clf_hidn, 2)
        ).to(self.device)

        self.policy = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.plcy_hidn),
            nn.ReLU(),
            nn.Linear(self.plcy_hidn, self.plcy_hidn),
            nn.ReLU(),
            nn.Linear(self.plcy_hidn, 1)
        ).to(self.device)

        self.temp = nn.Parameter(torch.tensor(0.0))          # Temperature parameter used by the heuristic policy

        self.gmm = None                                      # Container for the GMM fitted on the root space


        # Initialise Weights
        self.lift.apply(self.init_weights)
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.unlift.apply(self.init_weights)

        self.clf.apply(self.init_weights)
        self.policy.apply(self.init_weights)

    

    # Network weights initialisation
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

        


    # This procedure acts as a policy and outputs the action which the merger should take
    #   given the current token population (up tree) and iteration (iter)
    # The output shape is B x 2, where B is the micro-batch dimension, and the 2 dimensions
    #   are for the indeces in the up tree of which tokens are to be merged (in that order)
    def select_action(self, u_tree, iter):

        ########################## LOGITS MATRIX UPDATE #################################
        # First iteration builds the logit map for all of the tokens
        if iter == 0: 
            for i in range(self.n_inp_tk):
                for j in range(i + 1, self.n_inp_tk):

                    # The concatenated token pairs (in both orders)
                    true_x = torch.cat((
                        torch.cat((u_tree[:, i], u_tree[:, j]), dim=1),
                        torch.cat((u_tree[:, j], u_tree[:, i]), dim=1)
                    ), dim=0)

                    # The learned policy uses the logits predicted by the policy network
                    if self.policy_v == "learned":
                        logits = self.policy(true_x.detach()).flatten() 

                    # The heuristic policy uses the negative one-step reconstruction loss
                    else:
                        with torch.no_grad():
                            pred_x = self.decoder(self.encoder(true_x))
                            logits = -(torch.sum((pred_x - true_x) ** 2, dim=1) / (self.emb_dim * 2))

                    # Update the logits matrix
                    self.logits_mtx[:, i, j] = logits[:self.batch_dim] 
                    self.logits_mtx[:, j, i] = logits[self.batch_dim:]



        # Future iterations only compute for the new token
        else:
            nt_idx = self.n_inp_tk + iter - 1
            for i in range(self.active_tokens.size(1) - 1):

                # The concatenated token pairs (in both orders)
                true_x = torch.cat((
                    torch.cat((
                        u_tree[:, nt_idx], 
                        u_tree[torch.arange(self.batch_dim), self.active_tokens[:, i]]
                    ), dim=1),
                    torch.cat((
                        u_tree[torch.arange(self.batch_dim), self.active_tokens[:, i]],
                        u_tree[:, nt_idx]
                    ), dim=1)
                ), dim=0)

                # The learned policy uses the logits predicted by the policy network
                if self.policy_v == "learned":
                    logits = self.policy(true_x.detach()).flatten()

                # The heuristic policy uses the negative one-step reconstruction loss
                else:
                    with torch.no_grad():
                        pred_x = self.decoder(self.encoder(true_x))
                        logits = -(torch.sum((pred_x - true_x) ** 2, dim=1) / (self.emb_dim * 2))

                # Update the logits matrix
                self.logits_mtx[torch.arange(self.batch_dim), nt_idx, self.active_tokens[:, i]] = logits[:self.batch_dim]
                self.logits_mtx[torch.arange(self.batch_dim), self.active_tokens[:, i], nt_idx] = logits[self.batch_dim:]
        #################################################################################



        ########################## ENTROPY #################################
        # Decide if the temperature parameter is being used or not
        if self.policy_v == "learned":
            temp = 1
        else:
            temp = torch.exp(self.temp)    # Prevents the temperature from being <= 0

        # Build the probability matrix from the logits matrix
        #   We clone due to in-place operations (e.g. masking, new merges etc.) which inhibits gradient flow
        p_mtx = F.softmax(self.logits_mtx.clone().reshape(self.batch_dim, -1) / temp, dim=1)   
        
        # Adjust the output zeros from the softmax to avoid NaN propagation (from torch.special.xlogy gradients at 0)
        p_mtx_adj = (p_mtx + self.sftmx_eps) / (1 + (p_mtx.size(1) * self.sftmx_eps))

        # Used in adjusting the computed entropy
        val_norm = self.sftmx_eps / (1 + (p_mtx.size(1) * self.sftmx_eps))              # The (small) probability of unavailable actions (after sftmx_eps adjustment)
        opt_norm = self.active_tokens.size(1) * (self.active_tokens.size(1) - 1)        # The number of available actions
        nopt_norm = (p_mtx.size(1)) - opt_norm                                          # The number of unavailable actions

        # Compute and track the desired entropy of available actions only (normalised between 0 & 1)
        entrop = -(torch.sum(torch.special.xlogy(p_mtx_adj, p_mtx_adj), dim=1) - nopt_norm * (val_norm * math.log(val_norm))) / math.log(opt_norm)
        self.entropies += entrop
        ####################################################################

        
        ########################## SAMPLING & MASKING #################################
        with torch.no_grad():

            # Sample from the probability matrix
            sample_flat = torch.multinomial(p_mtx, 1)                
            sample = torch.cat((
                (sample_flat // self.logits_mtx.size(1)),
                (sample_flat % self.logits_mtx.size(1))
            ), dim=1)

            # Update the actions
            self.actions[:, iter] = sample

            # Masking the logits matrix & active tokens
            self.logits_mtx[torch.arange(self.batch_dim)[:, None], sample, :] = self.mask_val
            self.logits_mtx[torch.arange(self.batch_dim)[:, None], :, sample] = self.mask_val
            rm_ids = torch.cat((
                ((self.active_tokens - sample[:, 0, None]) == 0).nonzero()[:, 1, None],
                ((self.active_tokens - sample[:, 1, None]) == 0).nonzero()[:, 1, None]
            ), dim=1)   
            mask = torch.ones_like(self.active_tokens).scatter_(1, rm_ids, 0)
            self.active_tokens = self.active_tokens[mask.bool()].reshape((self.batch_dim, self.active_tokens.size(1) - 2))
        ###############################################################################


        
        # Track the one-step losses at the selected actions
        true_s = u_tree[torch.arange(self.batch_dim)[:, None], sample].reshape(self.batch_dim, -1).detach()
        pred_s = self.decoder(self.encoder(true_s))
        osl = torch.sum((pred_s - true_s) ** 2, dim=1) / (self.emb_dim * 2)      
        self.step_losses += osl


        # If using a learned policy, track the log probs and the rewards of the action
        if self.policy_v == "learned":
            self.log_probs[:, iter] = torch.log(p_mtx[torch.arange(self.batch_dim), sample_flat.flatten()])
            self.rewards[:, iter] = -osl.detach()

        # Return the action
        return sample




    # HTM forward pass
    #   x is the input: B x P x T, where B is micro batch dimension, P is token population
    #       size and T is the token dimensions
    #   merges denotes the number of encoder merges performed (equivalently, the number of
    #       actions taken). -1 is interpreted as full merging to the root space.
    def forward(self, x, merges=-1):

        # Reformat input into B x P x T
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        ########### Initialise Memory ###########
        self.batch_dim = x.size(0)
        self.n_inp_tk = x.size(1)
        self.step_losses = torch.zeros(self.batch_dim).to(self.device)
        self.entropies = torch.zeros(self.batch_dim).to(self.device)
        self.active_tokens = torch.arange(self.n_inp_tk).repeat(self.batch_dim, 1).to(self.device)

        # Clip to the maximum number of merges if applicable
        if (merges < 0) or (merges > self.n_inp_tk - 1):
            merges = self.n_inp_tk - 1
        
        self.actions = torch.zeros((self.batch_dim, merges, 2), dtype=torch.long).to(self.device)
        self.logits_mtx = torch.full((self.batch_dim, (self.n_inp_tk + merges - 1), (self.n_inp_tk + merges - 1)), self.mask_val).to(self.device)
        
        # Prepare classification results/labels containers if applicable
        if self.gen:
            clf_pred = torch.zeros((self.batch_dim, merges + self.n_inp_tk, 2)).to(self.device)
            clf_lbl = torch.cat((torch.ones(merges, dtype=torch.long), torch.zeros(self.n_inp_tk, dtype=torch.long))).repeat(self.batch_dim).reshape(self.batch_dim, -1).to(self.device)
        else:
            clf_pred = None
            clf_lbl = None

        # Prepare log_prob and rewards containers if applicable
        if self.policy_v == "learned":
            self.log_probs = torch.zeros((self.batch_dim, merges)).to(self.device)
            self.rewards = torch.zeros((self.batch_dim, merges)).to(self.device)
        #########################################



        # Lifting and adding noise
        u_tree = self.lift(x)
        u_tree = u_tree + self.noise_std * torch.randn_like(u_tree)


        # Iterative encoder merging
        for iter in range(merges):
            
            # Select action
            action = self.select_action(u_tree, iter)  

            # Merge and add noise         
            mg_tok = self.encoder(u_tree[torch.arange(self.batch_dim)[:, None], action].reshape(self.batch_dim, -1)).unsqueeze(1)
            mg_tok = mg_tok + self.noise_std * torch.randn_like(mg_tok)
            u_tree = torch.cat((u_tree, mg_tok), dim=1)

            # Update Active Tokens
            self.active_tokens = torch.cat((
                self.active_tokens, 
                torch.full((self.batch_dim, 1), self.n_inp_tk + iter).to(self.device)
            ), dim=1)

  

        # Transfer active tokens to the down tree
        d_tree = torch.zeros_like(u_tree)
        actk_key = (torch.arange(self.batch_dim)[:, None], self.active_tokens)
        d_tree[*actk_key] = u_tree[*actk_key]


        # Iterative decoder unmerging
        for iter in range(merges):

            # Obtain the merged token
            #   Note: We clone here because of the inplace operation from indexing a down_tree token
            #       back into the down_tree. Avoiding this may require a graphical data structure 
            #       instead of a tensor, but it may pose difficulties with vectorisation and autograd.
            key = -1 - iter
            mgd_tok = d_tree[:, key].clone() 

            # Unmerge and index into the down_tree according to the actions
            unmg_tok = self.decoder(mgd_tok).reshape(self.batch_dim * 2, -1)
            d_tree[torch.arange(self.batch_dim).repeat_interleave(2), self.actions[:, key].flatten()] = unmg_tok


            # Obtain classification result for merged token if applicable 
            if self.gen:       
                clf_pred[:, iter, :] = self.clf(mgd_tok.detach())

        # Classify the remaining leaves if applicable
        if self.gen:
            for i in range(self.n_inp_tk):
                clf_pred[:, -1 - i, :] = self.clf(d_tree[:, i].detach())


        # Adjust step-losses and entropy averages
        reinf_obj = None
        if merges > 0:
            self.step_losses /= merges
            self.entropies /= merges


            if self.policy_v == "learned":
                
                # Refresh the moving advantages container
                self.mvg_adv = torch.cat((self.mvg_adv, self.rewards.flatten()), dim=0)
                self.mvg_adv = self.mvg_adv[max(self.mvg_adv.size(0) - self.mvg_adv_sz, 0):]

                # Baseline the advantages for this episode (using normalisation)
                self.rewards = (self.rewards - self.mvg_adv.mean()) / (self.mvg_adv.std() + self.std_eps)

                # Build the reinforce objective (using immediate rewards as the advantages)
                reinf_obj = torch.sum((self.log_probs * self.rewards), dim=1)


        # Return: prediction, up_tree, down_tree, average step losses, average entropy, classifier prediction & labels, and the REINFORCE objective
        return (self.unlift(d_tree[:, :self.n_inp_tk]), u_tree, d_tree, self.step_losses, self.entropies, clf_pred, clf_lbl, reinf_obj)
        




    # Sample and generate a new (tokenized) image
    #   Note: This method is not vectorised due to possibly varying correct down_tree sizes
    def sample(self):

        # Raise an error if there is no fitted GMM for sampling the root spaces
        if self.gmm == None:
            raise Exception("Please fit a GMM model first...")

        steps = 0
        with torch.no_grad():

            # Initiate the down_tree with a GMM sample root token
            d_tree = torch.tensor(self.gmm.sample(1)[0], dtype=torch.float32).to(self.device)
        
            # Track the indeces in the down_tree corresponding to a leaf
            leaf_idx = []

            # Generation loop
            generate = True
            tok_idx = -1
            while generate:
                current_tok = d_tree[tok_idx]

                # Decide if the current token is merged or not
                clf_pred = self.clf(current_tok)
                sftmx = F.softmax(clf_pred, dim=0)
                thresh_mask = sftmx < self.decision_threshold
                decider = (thresh_mask == 0).nonzero()
                if decider.size(0) == 0:
                    is_mg = 0
                elif decider.size(0) == 2:
                    is_mg = 1
                else:
                    is_mg = decider.item()

                # If it is merged, decode it and enqueue it in the down_tree
                if is_mg:
                    unmgd_toks = self.decoder(current_tok).reshape(2, -1)
                    d_tree = torch.cat((unmgd_toks, d_tree), dim=0)
                
                # Otherwise, track the leaf index
                else:
                    leaf_idx.append(tok_idx)

                # Dequeue by reducing the index we check in the down_tree, and increase the 
                #   steps by 1
                tok_idx -= 1
                steps += 1

                # If we reach the end of the tree or the maximum step limit, stop the generation
                if (-1 * tok_idx) > d_tree.size(0) or (steps >= self.gen_step_limit):
                    generate = False


            # Return the unlifted leaves from the downtree
            leaves = d_tree[leaf_idx]
            leaves = self.unlift(leaves)
        return leaves
            


    # Returns a tensor of root tokens from a given dataset (num_rtx & batch_macro roughly
    #   control the size)
    # This is typically used to train the GMM for generation
    def sample_rts(self, ds, batch_macro, num_rtx=500):
        
        # Initiate the root tokens container and counter
        rtx = torch.empty((0, self.emb_dim)).to(self.device)
        samples_gen = 0

        # Prepare dataloader
        dl = DataLoader(ds, batch_size=batch_macro, shuffle=True, collate_fn=ds.collate_fn)

        # Sampling loop
        with torch.no_grad():
            while samples_gen < num_rtx:
                for batch in dl:
                    for tp in batch.keys():
                        inp_x = batch[tp].to(self.device)

                        # Forward pass to the root
                        _, u_tree, _, _, _, _, _, _ = self.forward(inp_x, merges=-1)

                        # Extract the last (root) token from the up_tree
                        rtx = torch.cat((rtx, u_tree[:, -1]), dim=0) 
                        samples_gen += inp_x.size(0)
                    if samples_gen >= num_rtx:
                        break
        return rtx



    ########################################################################################





















    ######################################### UTILITIES #########################################



    # Display the number of parameters of the networks
    def num_params(self):
        #print(f'> Total {len(torch.nn.utils.parameters_to_vector(self.parameters()))}')
        print("\t" + f'> Lifter: {len(torch.nn.utils.parameters_to_vector(self.lift.parameters()))}')
        print("\t" + f'> Encoder: {len(torch.nn.utils.parameters_to_vector(self.encoder.parameters()))}')
        print("\t" + f'> Decoder: {len(torch.nn.utils.parameters_to_vector(self.decoder.parameters()))}')
        print("\t" + f'> Unlifter: {len(torch.nn.utils.parameters_to_vector(self.unlift.parameters()))}')
        print("\n\t" + f'> Classifier: {len(torch.nn.utils.parameters_to_vector(self.clf.parameters()))}')
        print("\t" + f'> Policy: {len(torch.nn.utils.parameters_to_vector(self.policy.parameters()))}')
        print("\t> Temperature: 1")


    # Initialise the empty containers for tracking the training metrics and the model's
    #   hyperparameters / training parameters
    def init_log(self):

        # For the visualisations
        plot = {
            "g_x": [],
            "htm_loss": [],
            "recon_loss": [],
            "step_loss": [],
            "trans_loss": [],
            "clf_loss": [],

            "lift_infm": [],
            "enc_infm": [],

            "entropy_real": [],
            "entropy_target": [],

            "temperature": [],
            "rfc_loss": [],

            "lr_lifters": [],
            "lr_mergers": [],
            "lr_classif": [],
            "lr_policy": [],

            "tau": [],

            "merge_lns": [],
            

            "tok_dist": {},
            "micro_batch": [],

            "example_true": None,
            "example_pred": None,

            "ncomp": [],
            "aics": [],
            "bics": []

        }

        # Initialise with base hyperparameters
        hyper = {
            "inp_dim": self.inp_dim,
            "emb_dim": self.emb_dim,
            "mu_hidn": self.mu_hidn,
            "clf_hidn": self.clf_hidn,
            "plcy_hidn": self.plcy_hidn,
            "policy_v": self.policy_v,

            "noise_std": self.noise_std,
            "sftmx_eps": self.sftmx_eps,
            "mask_val": self.mask_val,
            "std_eps": self.std_eps,
            "mvg_adv_sz": self.mvg_adv_sz,
            "gen_step_limit": self.gen_step_limit,
            "decision_threshold": self.decision_threshold,
            "device": self.device
        }
        
        return (plot, hyper)



    # Visualise the model's training results
    #   <example> is used by the train() method to demonstrate the current reconstruction abilities
    #   <show_hyper> as True will display the model's hyperparameters as well
    #   <specific> can be set to (1-12) to zoom in on a specific visualisation for clarity
    def vis_log(self, example=None, show_hyper=False, specific=None):
        if specific == None:
            fig = plt.figure(figsize=(24,16))
            fig.suptitle("Training Log", fontsize=24)      
            plt.subplots_adjust(hspace=0.4) 
        else:
            fig = plt.figure(figsize=(24,12))
        

        # Build Example if given
        if example != None:
            with torch.no_grad():
                x_pred, _, _, _, _, _, _, _ = self.forward(example[0], merges=example[1])
            self.plot_log["example_true"] = (example[2](example[0]))
            self.plot_log["example_pred"] = (example[2](x_pred))
        
        # 3 or 6 - Example Reconstruction
        if self.plot_log["example_pred"] != None:
            # Ground Truth Image
            if (specific == None) or (specific in [3, 6]):
                if specific == None:
                    fig.add_subplot(4, 3, 3)
                else:
                    fig.add_subplot(1, 2, 1)
                plt.xlabel("Ground Truth")
                plt.title("Reconstruction Example")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(self.plot_log["example_true"][0], cmap=self.plot_log["example_true"][1])

            # Predicted Image
            if (specific == None) or (specific in [3, 6]):
                if specific == None:
                    fig.add_subplot(4, 3, 6)
                else:
                    fig.add_subplot(1, 2, 2)
                plt.xlabel("Predicted")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(self.plot_log["example_pred"][0], cmap=self.plot_log["example_pred"][1])
    

        # 1 - Reconstruction Losses
        if (specific == None) or (specific == 1):
            if specific == None:
                fig.add_subplot(4, 3, 1)
            plt.plot(self.plot_log["g_x"], self.plot_log["trans_loss"], "-", color="blueviolet", label="Transport")
            plt.plot(self.plot_log["g_x"], self.plot_log["step_loss"], "-", color="mediumvioletred", label="Step")
            plt.plot(self.plot_log["g_x"], self.plot_log["recon_loss"], "-", color="royalblue", label="Reconstr.")
            plt.plot(self.plot_log["g_x"], self.plot_log["htm_loss"], "--", color="steelblue", label="HTM (Total)")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.fill_between(x=[0, self.plot_log["g_x"][-1]], y1=0, y2=1e-1, color="yellow", interpolate=True, alpha=0.3)
            plt.fill_between(x=[0, self.plot_log["g_x"][-1]], y1=1e-1, y2=1, color="gold", interpolate=True, alpha=0.2)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Avg. Loss")
            plt.yscale("log")
            plt.legend()
            plt.title("Reconstruction Losses")


        # 2 - Entropy
        if (specific == None) or (specific == 2):
            if specific == None:
                fig.add_subplot(4, 3, 2)
            plt.plot(self.plot_log["g_x"], self.plot_log["entropy_real"], "-", color="salmon", label="Real")
            plt.plot(self.plot_log["g_x"], self.plot_log["entropy_target"], "-", color="sienna", label="Target")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Avg. Entropy")
            plt.legend()
            plt.title("Entropy")


        # 4 - Penalties
        if (specific == None) or (specific == 4):
            if specific == None:
                fig.add_subplot(4, 3, 4)
            plt.plot(self.plot_log["g_x"], self.plot_log["lift_infm"], "-", color="mediumseagreen", label="Lifter")
            plt.plot(self.plot_log["g_x"], self.plot_log["enc_infm"], "-", color="yellowgreen", label="Encoder")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Avg. Penalty")
            if self.hyper_log["reg"]:
                plt.yscale("log")         ######################################################################################### HERE #######################################
            else:                                                                           # likely need to do something similar for policy based on version flag though!
                plt.yscale("linear")  
            plt.legend()
            plt.title("Penalties")


        # 5 - Temperature or Reinforce Objective
        if (specific == None) or (specific == 5):
            if specific == None:
                fig.add_subplot(4, 3, 5)
            if self.hyper_log["policy_v"] != "learned":
                plt.plot(self.plot_log["g_x"], self.plot_log["temperature"], "-", color="salmon")
                plt.ylabel("Avg. Temperature")
                plt.title("Temperature (Exponent)")
            else:
                plt.axhline(y=0, color="thistle", linestyle='--')
                plt.plot(self.plot_log["g_x"], self.plot_log["rfc_loss"], "-", color="mediumvioletred")
                plt.ylabel("Avg. Objective Score")
                plt.title("Reinforce Objective")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
        

        # 7 - Classifier Loss
        if (specific == None) or (specific == 7):
            if specific == None:
                fig.add_subplot(4, 3, 7)
            plt.plot(self.plot_log["g_x"], self.plot_log["clf_loss"], "-", color="firebrick")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Avg. Loss")
            if self.hyper_log["gen"]:
                plt.yscale("log")
            else:
                plt.yscale("linear")
            plt.title("Classifier Loss")


        # 8 - LR Schedules
        if (specific == None) or (specific == 8):
            if specific == None:
                fig.add_subplot(4, 3, 8)
            lr_params = {
                "lifters": ["-", "dodgerblue" , "Lifters"],
                "mergers": ["--", "midnightblue", "Mergers"],
                "classif": ["--", "darkred", "Classifier"],
                "policy": ["-", "plum", "Policy"]
            }
            for nm in self.plot_log["opt_core_nms"]: 
                plt.plot(self.plot_log["g_x"], self.plot_log["lr_" + nm], lr_params[nm][0], color=lr_params[nm][1], label=lr_params[nm][2])
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Learning Rate")
            plt.yscale("log")
            plt.legend()
            plt.title("Learning Rate Schedules")


        # 9 - Micro Batch Dist.
        if (specific == None) or (specific == 9):
            if specific == None:
                fig.add_subplot(4, 3, 9)
            plt.bar(list(range(1, len(self.plot_log["micro_batch"]) + 1)), [i / sum(self.plot_log["micro_batch"]) for i in self.plot_log["micro_batch"]], color="burlywood")
            plt.xlim(right=(np.max(np.nonzero(self.plot_log["micro_batch"])).item()))
            plt.xlabel("Batch Dimension")
            plt.ylabel("Likelihood")
            plt.title("Micro Batch Distribution")


        # 10 - GMM tuning
        if (specific == None) or (specific == 10):
            if specific == None:
                fig.add_subplot(4, 3, 10)
            plt.plot(self.plot_log["ncomp"], self.plot_log["aics"], "-", color="coral", label="AIC")
            plt.plot(self.plot_log["ncomp"], self.plot_log["bics"], "-", color="indianred", label="BIC")
            plt.xlabel("Number of Components")
            plt.ylabel("Score")
            plt.legend()
            plt.title("GMM Best Fit vs Number of Components")
        

        # 11 - Tau schedule 
        if (specific == None) or (specific == 11):
            if specific == None:
                fig.add_subplot(4, 3, 11)
            plt.plot(self.plot_log["g_x"], self.plot_log["tau"], "-", color="blueviolet")
            for lns in self.plot_log["merge_lns"]:
                plt.axvline(x = lns, ls = "-", color = "black", alpha=0.3)
            plt.xlabel("Gradient Update Steps")
            plt.ylabel("Tau")
            plt.title("Tau Schedule")


        # 12 - Token Population Distribution
        if (specific == None) or (specific == 12):
            if specific == None:
                fig.add_subplot(4, 3, 12)
            keys = list(self.plot_log["tok_dist"].keys())
            keys.sort()
            ttl = 0
            for key in keys:
                ttl += self.plot_log["tok_dist"][key]
            plt.bar(keys, [self.plot_log["tok_dist"][i] / ttl for i in keys] , color="lightsalmon")
            plt.xlabel("Token Population")
            plt.ylabel("Likelihood")
            plt.title("Token Population Distribution")

        plt.show()

        # Display hyperparameters if applicable
        if show_hyper:
            print("\n\n\t\tHyperparameters\n")
            for key in self.hyper_log.keys():
                print(f"{key}: {self.hyper_log[key]}")



    # Save the HTM model parameters, the GMM model and the training logs
    def save(self, save_path="./train_results"):
        try:
            os.mkdir(save_path)
        except:
            pass

        # Save the logs and model parameters
        with open(save_path + "/HTM_meta_" + self.ufi + ".pkl", "wb") as log_f:
            pickle.dump((self.plot_log, self.hyper_log), log_f)
        with open(save_path + "/HTM_gmm_" + self.ufi + ".pkl", "wb") as log_f:
            pickle.dump(self.gmm, log_f)
        torch.save(self.state_dict(), save_path + "/HTM_model_" + self.ufi)
        


    # Load a trained model into this model
    #   Note that while it is possible to complete the transfer without reconstructing the 
    #       object for most properties, it is important to build the model with the correct
    #       network parameter dimensions.
    def load_model(self, ufi, path="./train_results"):
        with open(path + "/HTM_meta_" + ufi + ".pkl", "rb") as meta_f: 
            self.plot_log, self.hyper_log = pickle.load(meta_f)
        with open(path + "/HTM_gmm_" + ufi + ".pkl", "rb") as meta_f: 
            self.gmm = pickle.load(meta_f)

        self.load_state_dict(torch.load(path + "/HTM_model_" + ufi, weights_only=True))

        # Reassign variables
        self.ufi = ufi
        self.reg = self.hyper_log["reg"]
        self.gen = self.hyper_log["gen"]

        # Reassign Hyperparameters
        self.inp_dim = self.hyper_log["inp_dim"]
        self.emb_dim = self.hyper_log["emb_dim"]
        self.mu_hidn = self.hyper_log["mu_hidn"]
        self.clf_hidn = self.hyper_log["clf_hidn"]
        self.plcy_hidn = self.hyper_log["plcy_hidn"]
        self.policy_v = self.hyper_log["policy_v"]

        self.noise_std = self.hyper_log["noise_std"]
        self.sftmx_eps = self.hyper_log["sftmx_eps"]
        self.mask_val = self.hyper_log["mask_val"]
        self.std_eps = self.hyper_log["std_eps"]
        self.mvg_adv_sz = self.hyper_log["mvg_adv_sz"]
        self.gen_step_limit = self.hyper_log["gen_step_limit"]
        self.decision_threshold = self.hyper_log["decision_threshold"]
        self.device = self.hyper_log["device"]

        # Change device
        self.lift = self.lift.to(self.device)
        self.unlift = self.unlift.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.clf = self.clf.to(self.device)
        self.policy = self.policy.to(self.device)

        # Reset memory
        self.batch_dim = 0
        self.n_inp_tk = 0

        self.logits_mtx = None
        self.active_tokens = None
        self.actions = None

        self.step_losses = None
        self.entropies = None


        self.log_probs = None
        self.rewards = None
        self.mvg_adv = torch.empty(0).to(self.device)



    #############################################################################################

























    ######################################### TRAINING #########################################


    # Analyse the effect of changing the n_components used to fit the GMM model to the root spaces
    def tune_gmm(self, ds, batch_macro, num_rtx=500, ncomp_rnge=list(range(1, 21, 1)), cov_type="full", max_iter=100, tol=1e-3, live_plt=True, save=False, save_path = "./train_results"):
        
        # Obtain root space sample
        rtx = self.sample_rts(ds, batch_macro, num_rtx=num_rtx)
    
        # Reset plot containers
        self.plot_log["ncomp"] = []
        self.plot_log["aics"] = []
        self.plot_log["bics"] = []

        # Main tuning loop
        for i in range(len(ncomp_rnge)): 
            self.plot_log["ncomp"].append(ncomp_rnge[i])
            gmm = mixture.GaussianMixture(n_components=ncomp_rnge[i], covariance_type=cov_type, max_iter=max_iter, tol=tol)
            gmm.fit(rtx.cpu().numpy())

            # Measure the quality of the fit using BIC and AIC
            self.plot_log["aics"].append(gmm.aic(rtx.cpu().numpy()))
            self.plot_log["bics"].append(gmm.bic(rtx.cpu().numpy()))

            if live_plt:
                self.vis_log()  
                disp.clear_output(wait=True)
        
        # Track the changes
        self.hyper_log["tune_num_rtx"] = num_rtx
        self.hyper_log["tune_ncomp_rnge"] = ncomp_rnge
        self.hyper_log["tune_cov_type"] = cov_type
        self.hyper_log["tune_max_iter"] = max_iter
        self.hyper_log["tune_tol"] = tol

        # Resave
        if save:
            self.save(save_path)
            



    # Fit the GMM to the given root space sample data (rtx)
    def fit_gmm(self, rtx, ncomp=1, cov_type="full", max_iter=100, tol=1e-3, save=False, save_path = "./train_results"):
        self.gmm = mixture.GaussianMixture(n_components=ncomp, covariance_type=cov_type, max_iter=max_iter, tol=tol)
        self.gmm.fit(rtx.cpu().numpy())

        # Track the changes
        self.hyper_log["fit_num_rtx"] = len(rtx)
        self.hyper_log["fit_ncomp"] = ncomp
        self.hyper_log["fit_cov_type"] = cov_type
        self.hyper_log["fit_max_iter"] = max_iter
        self.hyper_log["fit_tol"] = tol

        # Resave
        if save:
            self.save(save_path)




    # The main HTM training pipeline
    def train(self, 
              train_ds,                                     # Training set (custom dataset object)
              batch_macro,                                  # Controls the effective (micro) batch dimension used in forward passes

              merges,                                       # The curriculum's merges (List)
              epochs,                                       # Epochs to train for at each curriculum/phase (List)


                 #Set these to None (List) if you wish to not use them

              lr_lu,                                        # (Un)Lifter learning rates across curriculum (List)
              lr_mu,                                        # ^^ but for (Un)Mergers
              lr_clf,                                       # ^^ but for Classifier
              lr_pol,                                       # ^^ but for Policy network
              lr_lu_decay,                                  # Epoch after which (Un)Lifters learning rates will start to linearly decay (List)
              lr_mu_decay,                                  # ^^ but for (Un)Mergers
              lr_clf_decay,                                 # ^^ but for Classifier
              lr_pol_decay,                                 # ^^ but for Policy network

              lmbda_lif,                                    # Lambda (List) for Lifter's penalty
              lmbda_enc,                                    # ^^ for Encoder
              lmbda_sl,                                     # Lambda (List) for step losses
              lmbda_trns,                                   # ^^ for transport losses

              gen = True,                                   # Are we training for the generative setting? (can be used to disable the classifier)
              policy_v = "learned",                         # Which policy version to use ("heuristic" or "learned")

              wrk_entrp = 0.3,                              # Working entropy      
              entrp_L = [0.05, 0.15, 0.2, 0.3],             # Entropy Target Schedule - proportion of total epochs with > 0 merges dedicated to: Sustain at 1, Decay to wrk_entrp, Decay to 0, and Sustain at 0
              lmbda_entrp = 3,                              # Lambda biasing the entropy loss in the REINFORCE objective (the lower the better and recommended to not exceed 5)
              
              tau_init = 0.5,                               # Initial Tau interpolating value
              tau_L = [0.05, 0.2],                          # Tau Schedule - proportion of total epochs with > 0 merges dedicated to: sustain at 1, and final sustain at 0
              
              lp = 1/100,                                   # Loss power (set to 0 or None if unused)
              plcy_l2 = 1e-5,                               # Weight decay (L2 regularisation) of Policy network
              plt_ivl = 100,                                # Plot interval (for visualisations)
              g_accum_f = 2,                                # Gradient accumulation factor
              lr_t = 0.05,                                  # Learning rate for the temperature optimiser in the heuristic setting

              sched_bnd = 1e-3,                             # Bound for the smallest lr decay factor
              entrp_eps = 0.0005,                           # Epsilon neighbourhood threshold for when the temperature optimiser is stuck at entropy of 1
              
              tune_num_rtx = 500,                           # Number of root samples used in the tuning of the GMM 
              tune_ncomp_rnge = list(range(1, 21, 1)),      # ^^, number of components analysed
              tune_cov_type = "full",                       # ^^, covariance type used
              tune_max_iter = 100,                          # ^^, max iterations of EM
              tune_tol = 1e-3,                              # ^^, tolerance threshold used in EM

              live_plt = True,                              # Render the training results live
              save = True,                                  # Save the model upon completion
              save_path = "./train_results",                # Directory in which to save the model
              print_policy = False                          # Display example actions taken during live plotting
              ):
        
        ###################### Prepare for new training ###########################
        # Prepare new logs 
        self.plot_log, self.hyper_log = self.init_log()
        self.plot_log["micro_batch"] = [0] * batch_macro

        hyperparams = locals()
        for arg in hyperparams.keys():
            if arg not in ["self", "train_ds", "live_plt", "save", "save_path"]:
                self.hyper_log[arg] = hyperparams[arg]
    
        self.reg = False
        self.hyper_log["reg"] = self.reg
        self.gen = gen
        self.policy_v = policy_v
        
        # Prepare dataloader
        data_loader = DataLoader(train_ds, batch_size=batch_macro, shuffle=True, collate_fn=train_ds.collate_fn)
        
        # Prepare optimisers
        opt_core_nms = []
        optim_params = []
        if self.policy_v == "learned":
            self.mvg_adv = torch.empty(0).to(self.device)
            opt_core_nms.append("policy")
            optim_params.append((opt_core_nms[-1], self.policy.parameters(), 0, lr_pol, lr_pol_decay))
        else:
            optim_params.append(("temp", [{"params": self.temp, "lr": lr_t}]))
        
        if self.gen:
            opt_core_nms.append("classif")
            optim_params.append((opt_core_nms[-1], self.clf.parameters(), 0, lr_clf, lr_clf_decay))
   
        opt_core_nms += ["lifters", "mergers"]
        optim_params += [
            (opt_core_nms[-2], (list(self.lift.parameters()) + list(self.unlift.parameters())), 0, lr_lu, lr_lu_decay),
            (opt_core_nms[-1], (list(self.encoder.parameters()) + list(self.decoder.parameters())), 0, lr_mu, lr_mu_decay),
        ]  
        self.plot_log["opt_core_nms"] = opt_core_nms

        optimisers = {}
        for i in range(len(optim_params)):
            if len(optim_params[i]) == 5:
                optimisers[optim_params[i][0]] = {
                    "opt": torch.optim.Adam(optim_params[i][1], lr=optim_params[i][2], weight_decay=(plcy_l2 if (optim_params[i][0] == "policy") else 0)),
                    "lrs": optim_params[i][3],
                    "dec": optim_params[i][4]
                }
            else:
                optimisers[optim_params[i][0]] = {
                    "opt": torch.optim.Adam(optim_params[i][1]),
                }     
        del(optim_params)
        del(opt_core_nms)   
        

        # Prepare criterions
        rec_crit = nn.MSELoss() 
        entrp_crit = nn.L1Loss()
        clf_crit = nn.CrossEntropyLoss()

        # Prepare tracking variables
        g_upd_steps = 0
        g_accum_steps = 0

        htm_loss_accum = 0
        recon_loss_accum = 0
        step_loss_accum = 0
        trans_loss_accum = 0
        lift_infm_accum = 0
        enc_infm_accum = 0
        entropy_accum = 0
        entrp_trg_accum = 0
        temp_accum = 0
        clf_loss_accum = 0
        rfc_loss_accum = 0
        
        # Prepare scheduling variables
        sched_steps = (len(train_ds) * (sum(epochs) - ((merges[0] == 0) * epochs[0]))) / (batch_macro * g_accum_f)
        sched_steps_offset = (len(train_ds) * ((merges[0] == 0) * epochs[0])) / (batch_macro * g_accum_f)

        entrp_trg = 1
        entrp_disable = False
        xplr_dec_step = sched_steps_offset + entrp_L[0] * sched_steps
        wrk_start_step = xplr_dec_step + entrp_L[1] * sched_steps
        xplt_dec_step = wrk_start_step + (1 - sum(entrp_L)) * sched_steps
        xplt_end_step = xplt_dec_step + entrp_L[2] * sched_steps


        if tau_init != None:
            tau = tau_init
            tau_start = (tau_L[0] * sched_steps) + sched_steps_offset
            tau_end = tau_start + sched_steps * (1 - sum(tau_L))
        else:
            tau = 0

        # Initialise the max merges as 1 for the loss factor for now
        max_merges = 1

        # Train through the curriculums
        for ccm in range(len(merges)):
            
            ###################### Prepare for new curriculum ###########################
            epoch = 1
            self.plot_log["merge_lns"].append(g_upd_steps)
            g_upd_offset = g_upd_steps
            g_upd_exp = (len(train_ds) * epochs[ccm]) / (batch_macro * g_accum_f)
            

            # Prepare optimisers
            for nm in optimisers.keys():
                if nm == "temp":
                    optimisers[nm]["opt"].zero_grad()
                elif optimisers[nm]["lrs"][ccm] != None:
                    optimisers[nm]["opt"].param_groups[0]["lr"] = optimisers[nm]["lrs"][ccm]
                    optimisers[nm]["opt"].zero_grad()

            # Prepare LR Schedules
            for nm in optimisers.keys():
                if nm != "temp":
                    if optimisers[nm]["dec"][ccm] != None:
                        optimisers[nm]["step"] = (len(train_ds) * optimisers[nm]["dec"][ccm]) / (batch_macro * g_accum_f)
                    else:
                        optimisers[nm]["step"] = False

            # Update loss factor denominator
            if (merges[ccm] < 0) or (merges[ccm] > max_merges):
                denom = max_merges
            elif merges[ccm] == 0:
                denom = 1
            else:
                denom = merges[ccm]

            ########################################################################

            # Epoch training loop
            while epoch <= epochs[ccm]:
                for batch in data_loader:

                    # Simple normalising factor using the number of different token
                    #   populations in this batch
                    snf = len(batch.keys())
                    for tp in batch.keys():
                        x_train = batch[tp].to(self.device)

                        # Forward Trace
                        x_pred, u_t, d_t, step_losses, entropies, clf_pred, clf_lbl, rfc_obj = self.forward(x_train, merges=merges[ccm])

                        # End-to-End Reconstruction Loss
                        recon_loss = rec_crit(x_pred, x_train)
                        recon_loss_accum += recon_loss.item() / (snf * g_accum_f)

                        # One Step Losses
                        if (lmbda_sl[ccm] != None) and (merges[ccm] != 0):
                            step_loss = lmbda_sl[ccm] * torch.mean(step_losses)
                            step_loss_accum += step_loss.item() / (snf * g_accum_f * lmbda_sl[ccm])
                        else:
                            step_loss = 0
                        
                        # Lifter / Encoder Penalties
                        if lmbda_lif[ccm] != None:
                            lift_infm = u_t[:, :x_train.size(1)]
                            lift_penalty = lmbda_lif[ccm] * torch.mean((lift_infm ** 2))
                            lift_infm_accum += lift_penalty.item() / (snf * g_accum_f * lmbda_lif[ccm])
                        else:
                            lift_penalty = 0
                        if (lmbda_enc[ccm] != None) and (merges[ccm] != 0):
                            enc_infm = u_t[:, x_train.size(1):]
                            enc_penalty = lmbda_enc[ccm] * torch.mean((enc_infm ** 2))
                            enc_infm_accum += enc_penalty.item() / (snf * g_accum_f * lmbda_enc[ccm])
                        else:
                            enc_penalty = 0


                        # Transport Loss
                        if (lmbda_trns[ccm] != None) and (merges[ccm] != 0):
                            trans_loss = lmbda_trns[ccm] * torch.mean(((((1 - tau) * d_t) + ((tau - 1) * u_t.detach())) ** 2))   
                            trans_loss_accum += trans_loss.item() / (snf * g_accum_f * lmbda_trns[ccm])
                        else:
                            trans_loss = 0

                        
                        # HTM Loss Function
                        loss_factor = min((x_train.size(1) - 1) / denom, 1) ** (lp)
                        loss = loss_factor * (recon_loss + step_loss + trans_loss + lift_penalty + enc_penalty)
                        loss /= (snf * g_accum_f)
                        loss.backward()
                        htm_loss_accum += loss.item() / loss_factor


                        # Entropy Optimisation, (including REINFORCE)
                        if merges[ccm] != 0:
                            entropy_loss = entrp_crit(entropies, (entrp_trg * torch.ones_like(entropies))) 
                            
                            if self.policy_v == "learned":
                                if entrp_disable:
                                    entropy_loss = 0
                                rfc_loss = torch.mean(-rfc_obj)
                                pol_loss = rfc_loss + lmbda_entrp * entropy_loss 
                                pol_loss /= (snf * g_accum_f)
                                pol_loss.backward()
                                rfc_loss_accum += rfc_loss.item() / (snf * g_accum_f)
                                
                            else:
                                entropy_loss /= (snf * g_accum_f)
                                if not entrp_disable:
                                    entropy_loss.backward()
                                temp_accum += self.temp.item() / (snf * g_accum_f)
                            

                            entropy_accum += torch.mean(entropies.detach()).item() / (snf * g_accum_f)
                            entrp_trg_accum += entrp_trg / (snf * g_accum_f)
                            

                        # Classifier Training
                        if self.gen:
                            clf_loss = clf_crit(clf_pred.reshape(-1, 2), clf_lbl.flatten())
                            clf_loss /= (snf * g_accum_f)
                            clf_loss.backward()
                            clf_loss_accum += clf_loss.item()

                        # Track Token Distribution
                        if (ccm == 0) and (epoch == 1):
                            if tp in self.plot_log["tok_dist"]:
                                self.plot_log["tok_dist"][tp] += x_train.size(0)
                            else:
                                self.plot_log["tok_dist"][tp] = x_train.size(0)

                        # Track Micro Batch Distribution
                        self.plot_log["micro_batch"][x_train.size(0) - 1] += 1


                    # Accumulate Gradients
                    g_accum_steps += 1

                    if g_accum_steps % g_accum_f == 0:
                        
                        # Step Optimisers
                        if optimisers["lifters"]["lrs"][ccm] != None:
                            optimisers["lifters"]["opt"].step()
                            optimisers["lifters"]["opt"].zero_grad()
                        if optimisers["mergers"]["lrs"][ccm] != None:
                            optimisers["mergers"]["opt"].step()
                            optimisers["mergers"]["opt"].zero_grad()
                        if self.gen and (optimisers["classif"]["lrs"][ccm] != None):
                            optimisers["classif"]["opt"].step()
                            optimisers["classif"]["opt"].zero_grad()
                        if (merges[ccm] != 0):
                            if self.policy_v == "learned":
                                optimisers["policy"]["opt"].step()
                                optimisers["policy"]["opt"].zero_grad()
                            elif not entrp_disable:
                                optimisers["temp"]["opt"].step()
                                optimisers["temp"]["opt"].zero_grad()

                        g_upd_steps += 1

                        # Update LR Schedule
                        for nm in self.plot_log["opt_core_nms"]:
                            if optimisers[nm]["step"]:
                                optimisers[nm]["opt"].param_groups[0]["lr"] = optimisers[nm]["lrs"][ccm] * max((1.0 - max(0, g_upd_steps - g_upd_offset - optimisers[nm]["step"]) / (g_upd_exp - optimisers[nm]["step"])), sched_bnd)

                        # Update Entropy Target
                        if g_upd_steps > xplr_dec_step:
                            entrp_trg = wrk_entrp
                            if g_upd_steps < wrk_start_step:
                                entrp_trg = wrk_entrp + ((1 - wrk_entrp) * max((1.0 - ((g_upd_steps - xplr_dec_step) / (wrk_start_step - xplr_dec_step + sched_bnd))), sched_bnd))
                            if g_upd_steps >= xplt_dec_step:
                                entrp_trg = wrk_entrp * max((1.0 - ((g_upd_steps - xplt_dec_step) / (xplt_end_step - xplt_dec_step + sched_bnd))), entrp_eps)


                        # Update Tau
                        if (tau_init != None):
                            tau = tau_init * max((1.0 - max(0, g_upd_steps - tau_start) / (tau_end - tau_start + sched_bnd)), 0)


                        # Track Data
                        if g_upd_steps % plt_ivl == 0:

                            self.plot_log["g_x"].append(g_upd_steps)
                            self.plot_log["htm_loss"].append(htm_loss_accum / plt_ivl)
                            self.plot_log["recon_loss"].append(recon_loss_accum / plt_ivl)

                            self.plot_log["clf_loss"].append(self.logger_fn(self.gen, clf_loss_accum / plt_ivl))
                            self.plot_log["rfc_loss"].append(self.logger_fn(((self.policy_v == "learned") and (merges[ccm] != 0)), rfc_loss_accum / plt_ivl))

                            
                            
                            self.plot_log["lift_infm"].append(self.logger_fn(lmbda_lif[ccm], lift_infm_accum / plt_ivl))
                            self.plot_log["enc_infm"].append(self.logger_fn(not ((lmbda_enc[ccm] == None) or (merges[ccm] == 0)), enc_infm_accum / plt_ivl))
                            if not self.reg:
                                if (self.plot_log["lift_infm"][-1] != None) or (self.plot_log["enc_infm"][-1] != None):
                                    self.reg = True
                                    self.hyper_log["reg"] = self.reg


                            self.plot_log["step_loss"].append(self.logger_fn(not ((lmbda_sl[ccm] == None) or (merges[ccm] == 0)), step_loss_accum / plt_ivl))
                            self.plot_log["trans_loss"].append(self.logger_fn(not ((lmbda_trns[ccm] == None) or (merges[ccm] == 0)), trans_loss_accum / plt_ivl))
                            self.plot_log["entropy_real"].append(self.logger_fn((merges[ccm] != 0), entropy_accum / plt_ivl)) 
                            self.plot_log["temperature"].append(self.logger_fn(((self.policy_v != "learned") and (merges[ccm] != 0)), temp_accum / plt_ivl)) 
                            self.plot_log["entropy_target"].append(self.logger_fn((merges[ccm] != 0), entrp_trg_accum / plt_ivl)) 
                            self.plot_log["tau"].append(self.logger_fn(not ((tau_init == None) or (merges[ccm] == 0)), tau))

                            for nm in self.plot_log["opt_core_nms"]:
                                self.plot_log["lr_" + nm].append(self.logger_fn(optimisers[nm]["lrs"][ccm], optimisers[nm]["opt"].param_groups[0]["lr"]))


                            # Reset the temperature optimiser once it gets stuck at entropy of 1
                            if (self.policy_v != "learned") and (merges[ccm] != 0) and (entrp_trg != 1) and (abs(1 - (entropy_accum / plt_ivl)) < entrp_eps): 
                                optimisers["temp"]["opt"] = torch.optim.Adam([{"params": self.temp, "lr": lr_t}]) 
                            
                            # Toggle between allowing the temperature optimiser to update the temperature or not
                            #   (prevents the temperature from becoming too small)
                            if (ccm == len(merges) - 1) and (merges[ccm] != 0) and (abs((entropy_accum / plt_ivl) - 0) < (wrk_entrp * entrp_eps)):
                                entrp_disable = True
                            if entrp_disable and (abs((entropy_accum / plt_ivl) - 0) > (wrk_entrp * entrp_eps)):
                                entrp_disable = False
                            

                            # Plot Data
                            if live_plt:
                                if print_policy:
                                    print(">Policy Example:\n", self.actions[-1], "\n>Shape: ", self.actions.shape)
                                self.vis_log(example=(batch[max(batch.keys())][-1].to(self.device), merges[ccm], train_ds.visualise))
                                disp.clear_output(wait=True)
                                
                            # Reset Trackers
                            htm_loss_accum = 0
                            recon_loss_accum = 0
                            step_loss_accum = 0
                            trans_loss_accum = 0
                            lift_infm_accum = 0
                            enc_infm_accum = 0
                            entropy_accum = 0  
                            entrp_trg_accum = 0   
                            temp_accum = 0  
                            clf_loss_accum = 0
                            rfc_loss_accum = 0         

                # Update loss factor denominator and max_merges after the very first epoch
                if (ccm == 0) and (epoch == 1):
                    options = list(self.plot_log["tok_dist"].keys())
                    options.sort()
                    max_merges = int(options[-1]) - 1
                    if (merges[ccm] < 0) or (merges[ccm] > max_merges):
                        denom = max_merges
                    elif merges[ccm] == 0:
                        denom = 1
                    else:
                        denom = merges[ccm]
                
                epoch += 1

            

        


        ####################### After HTM Training #####################


        # GMM Tuning 
        if self.gen:
            self.tune_gmm(train_ds, batch_macro, num_rtx=tune_num_rtx, ncomp_rnge=tune_ncomp_rnge, cov_type=tune_cov_type, max_iter=tune_max_iter, tol=tune_tol, live_plt=live_plt)


            

        # Save the model with its training results if applicable
        if save:
            self.save(save_path)
            
        

    ############################################################################################       

        