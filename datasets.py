import random
import math
import cv2

from scipy.spatial import KDTree
from PIL import Image, ImageDraw


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from einops import rearrange


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Custom dataset consisting of crosses
#   A cross structure is composed of 2 tokens, where a token is a 4D representation of 
#       a line segment: (x1, y1, x2, y2)
#   Crosses contain line segments of the same length, and cross at their midpoint
class Crosses_DS(Dataset):

    # Constructor
    def __init__(self,
                 ds_size,                     # The size of the dataset
                 line_len = (10, 10),         # The (possible) length of each of the line segments in a cross
                 num_crosses = [1, 2, 3],     # The number of crosses that could appear in an image
                 num_crss_dist = "uniform",   # Weightings for the likelihood of each num. of crosses ("uniform" gives a uniform dist.)
                 img_size = (128, 128),       # Size of the images
                 angular_sep = (20, 160)      # The range of angles separating the line segments at a cross
                 ):
        
        # Properties
        self.ds_size = ds_size
        self.line_len = line_len
        self.num_crosses = num_crosses
        self.num_crss_dist = num_crss_dist
        self.img_size = img_size
        self.angular_sep = angular_sep
        self.cmap = "gray"

        # Generate the data (stored in memory)
        self.ds = self.generate_data()


    # Get item from dataset
    def __getitem__(self, idx):
        return self.ds[idx]

    # Length of dataset
    def __len__(self):
        return len(self.ds)
    
    
    # Generate the tokenized data
    def generate_data(self):

        # Prepare dataset container
        ds = []

        # Prepare the weights controlling the token population distribution
        if self.num_crss_dist == "uniform":
            wghts = (1 / len(self.num_crosses)) * torch.ones(len(self.num_crosses))
        else:
            wghts = torch.tensor(self.num_crss_dist, dtype=torch.float)

        # Generation loop
        for i in range(self.ds_size):
            tokens = []

            # Sample token population from the distribution
            for j in range(self.num_crosses[torch.multinomial(wghts, 1).item()]):

                # Initiate random line segment parameters
                cx = random.random() * self.img_size[0]
                cy = random.random() * self.img_size[1] 
                theta1 = math.radians(random.randint(0, 359))
                theta2 = theta1 + math.radians(random.randint(*self.angular_sep))
                line_l = random.randint(*self.line_len)

                # Build line segment pairs (crosses)
                x1 = cx + (line_l / 2) * math.cos(theta1)
                y1 = cy + (line_l / 2) * math.sin(theta1)
                x2 = cx - (line_l / 2) * math.cos(theta1)
                y2 = cy - (line_l / 2) * math.sin(theta1)

                x3 = cx + (line_l / 2) * math.cos(theta2)
                y3 = cy + (line_l / 2) * math.sin(theta2)
                x4 = cx - (line_l / 2) * math.cos(theta2)
                y4 = cy - (line_l / 2) * math.sin(theta2)

                # Save the tokens
                token = [[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                        [round(x3, 4), round(y3, 4), round(x4, 4), round(y4, 4)]]
                tokens.append(token)
            ds.append(tokens)
        return ds
    


    # Used by the HTM for loading batches of data
    #   The output is a dictionary, where the keys are integer token populations
    #       and the values are tensors of dimension: B x P x T, where B is micro-batch
    #       dimension, P is token population and T is token dimension
    def collate_fn(self, batch):
        batch_dict = {}
        for i in range(len(batch)):
            key = len(batch[i]) * 2 
            entry_data = []
            for j in range(len(batch[i])):
                entry_data.append(batch[i][j][0])
                entry_data.append(batch[i][j][1])
            entry = torch.tensor(entry_data).unsqueeze(0)
            if key in batch_dict:
                batch_dict[key] = torch.cat((batch_dict[key], entry))
            else:
                batch_dict[key] = entry
        return batch_dict


    

    # Visualises the images from tokens, one image at a time (due to
    #   possible varying P dim)
    #   
    #   pil = True can be used to display the output images directly    
    #   
    #   NOTE: It is important that this method can visualise the images from any
    #       number of tokens, since the sampling may produce an unexpected number of them
    def visualise(self, data, pil=False):
        if type(data) == torch.Tensor:
            if len(data.shape) == 3:
                data = data.squeeze(0)
            data = data.unsqueeze(1)
        
        # Use PIL's ImageDraw to visualise the images from token data
        img = Image.new("L", size=self.img_size, color=255)
        draw = ImageDraw.Draw(img)
        for j in range(len(data)):
            draw.line([data[j][0][0], data[j][0][1], data[j][0][2], data[j][0][3]], width=1)
            if type(data) != torch.Tensor:
                draw.line([data[j][1][0], data[j][1][1], data[j][1][2], data[j][1][3]], width=1)
        if pil:
            return img
        else:
            return (np.array(img), self.cmap)







# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #





# Custom dataset consisting of the nodes from binary trees
#   Tokens are 2D points (x1, y1)
#   The separation between children and parent nodes throughout the height of the tree
#       is controlled by a constant scale factor, angle of separation and angle of deviation
class Trees_DS(Dataset):

    # Constructor
    def __init__(self,
                 ds_size,                            # The size of the dataset
                 sep_angle = 45,                     # The angle: left-child, parent, right-child
                 dev_angle = 20,                     # The angle by which children subtrees deviate
                 dist_factor = 0.5,                  # The scale factor shrinking the distance from parent to child
                 
                 num_trees = [1],                    # The number of trees that could appear in an image
                 num_trs_dist = "uniform",           # Weightings for the likelihood of each num. of trees ("uniform" gives a uniform dist.)
                 heights = [2, 3, 4],                # The possible tree heights
                 heights_dist = "uniform",           # Weightings for the likelihood of each height ("uniform" gives a uniform dist.)
                 init_dist = 50,                     # The initial distance from the root node to its children
                 centre_factor = 1/4,                # The proportion of the image that is considered the centre zone where roots spawn (for visibility)
                 point_radius = 1.5,                 # The radius of the points representing the node tokens when visualised
                 img_size = (256, 256)               # Size of the images 

                 ):
        
        # Properties
        self.ds_size = ds_size
        self.sep_angle = sep_angle
        self.dev_angle = dev_angle
        self.dist_factor = dist_factor

        self.num_trees = num_trees
        self.num_trs_dist = num_trs_dist
        self.heights = heights
        self.heights_dist = heights_dist
        self.init_dist = init_dist
        self.centre_factor = centre_factor
        self.point_radius = point_radius
        self.img_size = img_size
        self.cmap = "gray"

        # Generate the data (stored in memory)
        self.ds = self.generate_data()


    # Get item from dataset
    def __getitem__(self, idx):
        return self.ds[idx]


    # Length of dataset
    def __len__(self):
        return len(self.ds)



    # Recursive function used to generate children subtrees given the parent node's:
    #   position, height, angle and distance, where angle denotes the line of symmetry between
    #   the children subtrees.
    # NOTE: The resulting data format is [parent node, left-subtree, right-subtree], where each subtree 
    #   has the same data format until leaves are reached.
    def build_tree(self, node, height, angle, dist):
        if height == 1:
            return [node]
        else:
            # Calculate angular displacement for the left/right children nodes
            thetaL = angle + (0.5 * math.radians(self.sep_angle))
            thetaR = angle - (0.5 * math.radians(self.sep_angle))

            # Calculate the children node's positions (tokens)
            lnx = node[0] + (dist * math.cos(thetaL))
            lny = node[1] + (dist * math.sin(thetaL))
            rnx = node[0] + (dist * math.cos(thetaR))
            rny = node[1] + (dist * math.sin(thetaR))

            # Decrease the height and the distance for the next subtrees
            height -= 1
            dist *= self.dist_factor

            # Recursive step, including the angle of deviation
            return [node] + self.build_tree([lnx, lny], height, thetaL + math.radians(self.dev_angle), dist) + self.build_tree([rnx, rny], height, thetaR - math.radians(self.dev_angle), dist)



    # Generate the tokenized data
    def generate_data(self):

        # Prepare dataset container
        ds = []

        # Prepare the weights controlling the token population distribution
        if self.num_trs_dist == "uniform":
            num_wghts = (1 / len(self.num_trees)) * torch.ones(len(self.num_trees))
        else:
            num_wghts = torch.tensor(self.num_trs_dist, dtype=torch.float)
        if self.heights_dist == "uniform":
            hght_wghts = (1 / len(self.heights)) * torch.ones(len(self.heights))
        else:
            hght_wghts = torch.tensor(self.heights_dist, dtype=torch.float)

        # Generation loop
        for i in range(self.ds_size):
            tokens = []

            # Sample token population from the distributions and generate random root parameters
            for j in range(self.num_trees[torch.multinomial(num_wghts, 1).item()]):
                rtx = (self.img_size[0] * ((1-self.centre_factor)/2)) + (random.random() * self.img_size[0] * self.centre_factor)
                rty = (self.img_size[1] * ((1-self.centre_factor)/2)) + (random.random() * self.img_size[1] * self.centre_factor)
                h = self.heights[torch.multinomial(hght_wghts, 1).item()]
                ang = math.radians(random.randint(0, 359))
                
                # Build the tree data and save the tokens (positions)
                data = self.build_tree([rtx, rty], h, ang, self.init_dist)
                tokens.append(data)
            ds.append(tokens)
        return ds
    

    # Used by the HTM for loading batches of data
    #   The output is a dictionary, where the keys are integer token populations
    #       and the values are tensors of dimension: B x P x T, where B is micro-batch
    #       dimension, P is token population and T is token dimension
    def collate_fn(self, batch):
        batch_dict = {}
        for i in range(len(batch)):            
            entry_data = []
            for tree in range(len(batch[i])):
                for tk in range(len(batch[i][tree])):
                    entry_data.append(batch[i][tree][tk])
            key = len(entry_data)
            entry = torch.tensor(entry_data).unsqueeze(0)
            if key in batch_dict:
                batch_dict[key] = torch.cat((batch_dict[key], entry))
            else:
                batch_dict[key] = entry
        return batch_dict



    # Uses PIL's ImageDraw to visualise the token (node) as a circle 
    def draw_point(self, data, drawObj):
        drawObj.ellipse((
            data[0] - self.point_radius,
            data[1] - self.point_radius,
            data[0] + self.point_radius,
            data[1] + self.point_radius
        ), fill=0)





    # Visualises the images from tokens, one image at a time (due to
    #   possible varying P dim)
    #   
    #   pil = True can be used to display the output images directly    
    #   
    #   NOTE: It is important that this method can visualise the images from any
    #       number of tokens, since the sampling may produce an unexpected number of them
    def visualise(self, data, pil=False):
        img = Image.new("L", size=self.img_size, color=255)
        draw = ImageDraw.Draw(img)
        if type(data) == torch.Tensor:
            if len(data.shape) == 3:
                data = data.squeeze(0)
            for i in range(len(data)):
                self.draw_point(data[i], draw) 
        else:
            # Extra for loop due to number-of-trees dimension
            for i in range(len(data)):
                for j in range(len(data[i])):
                    self.draw_point(data[i][j], draw)
        if pil:
            return img
        else:
            return (np.array(img), self.cmap)






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



# Custom dataset for MNIST data
class MNIST_DS(Dataset):

    # Constructor
    def __init__(self,
                 root = "pytorch_datasets",      # The folder containing the downloaded dataset
                 train = True,                   # If True, then the training set is used, otherwise the testing set is used
                 threshv = 130,                  # The intensity threshold value (0-255)
                 ):
        
        # Properties
        self.root = root
        self.train = train
        self.threshv = threshv / 255
        self.cmap = "gray"

        if self.train:
            self.ds_type = "Training Set"
        else:
            self.ds_type = "Testing Set"
        
        self.ds = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=True,
            transform=ToTensor()
        )
        
        # Prepare KDTree for nearest-neighbour untokenization
        self.grid = np.concat((
            np.arange(28).repeat(28)[:, None],
            np.tile(np.arange(28), 28)[:, None]
        ), axis=1)
        self.tree = KDTree(self.grid) 


    # Get item from dataset
    def __getitem__(self, idx):
        return self.ds[idx]

    # Length of dataset
    def __len__(self):
        return len(self.ds)


    # Tokenize the MNIST Image
    #   Tokens are 2D indeces of pixels in the image that have an intensity higher than
    #       the threshold value.
    def tokenize(self, img):
        return (img.reshape(28, 28) >= self.threshv).nonzero().type(torch.float)
    


    # Untokenize the MNIST tokens back into an image
    #   Start with a black background, then assign white pixels based on the nearest
    #       gridpoint to each token input
    def untokenize(self, inp_tks):
        out_img = torch.zeros((28, 28))
        _, idx = self.tree.query(inp_tks.cpu().numpy())
        nn_idx = torch.tensor(self.grid[idx])
        out_img[nn_idx[:, 0], nn_idx[:, 1]] = 1
        return out_img


    # Used by the HTM for loading batches of data
    #   The output is a dictionary, where the keys are integer token populations
    #       and the values are tensors of dimension: B x P x T, where B is micro-batch
    #       dimension, P is token population and T is token dimension
    def collate_fn(self, batch):
        batch_dict = {}
        for i in range(len(batch)):   
            tks = self.tokenize(batch[i][0])
            key = len(tks)
            if key in batch_dict:
                batch_dict[key] = torch.cat((batch_dict[key], tks.unsqueeze(0)))
            else:
                batch_dict[key] = tks.unsqueeze(0)
        return batch_dict

    


    # Visualises the images from tokens or image data, one image at a time (due to
    #   possible varying P dim)
    #   
    #   direct = True can be used to display the output images directly    
    #   
    #   NOTE: It is important that this method can visualise the images from any
    #       number of tokens, since the sampling may produce an unexpected number of them
    def visualise(self, data, direct=False):
        if type(data) == tuple:
            vis = data[0].reshape(28, 28)
            if direct:
                plt.imshow(vis, cmap=self.cmap)
            else:
                return (vis, self.cmap)
        else:
            if data.size() == torch.Size([1, 28, 28]):
                vis = data.reshape(28, 28)
                if direct:
                    plt.imshow(vis, cmap=self.cmap)
                else:
                    return (vis, self.cmap)
                
            else:
                if len(data.shape) == 3:
                    data = data.squeeze(0)
                img = self.untokenize(data)
                if direct:
                    plt.imshow(img, cmap=self.cmap)
                else:
                    return (img, self.cmap)   # HERE, np.array instead??


    

    # We use Otsu Thresholding to help guide the choice of what threshold value to 
    #   use (on average) across all images. Binary thresholding using this value is
    #   more efficient to compute.
    def otsu_analysis(self):

        # Prepare the containers storing the otsu recommended threshold values and
        #   the intensity histograms
        otsu_threshs = []
        hist_cuml = np.zeros((256, 1))

        # Collect the data
        for i in range(len(self.ds)):
            img, _ = self.ds[i]
            img = (img*255).reshape(28, 28, 1).numpy().astype(np.uint8)
            hist_cuml += cv2.calcHist([img], [0], None, [256], [0, 256])
            otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            otsu_threshs.append(otsu_thresh)
        hist_cuml /= len(self.ds)
        bckts = np.arange(0, 256)
        otsu_ndarr = np.array(otsu_threshs)

        # Statistics
        mean = otsu_ndarr.mean()
        std = otsu_ndarr.std()
        left_sig = mean - std
        right_sig = mean + std
        less_mean = sum(otsu_ndarr < mean) / len(self.ds)
        high_mean = sum(otsu_ndarr >= mean) / len(self.ds)
        
        # Visualise the results
        fig = plt.figure(figsize=(16, 8))
        fig.add_subplot(1, 2, 1)
        plt.hlines(1, 0, 256, color="gray", ls=":", alpha=0.5)
        plt.plot(hist_cuml, alpha=0)
        plt.vlines(mean, 0, 10, color="indianred", ls="--")
        plt.fill_between(bckts, 0, hist_cuml[:, 0], color="indianred", alpha=0.3)
        plt.fill_between(bckts, 0, hist_cuml[:, 0], where=(bckts >= left_sig) & (bckts <= right_sig), interpolate=True, color="indianred", alpha=0.4)
        plt.xlabel("Intensity")
        plt.ylabel("Number of Pixels")
        plt.yscale("log")
        plt.vlines([60, 130], 0, 1, color="gray", ls="--", alpha=0.7)
        ints_vis = [
            Patch(facecolor=(0, 0, 0), edgecolor="indianred", label="0"),
            Patch(facecolor=(60/255, 60/255, 60/255), edgecolor="indianred", label="60"),
            Patch(facecolor=(130/255, 130/255, 130/255), edgecolor="indianred", label="130"),
            Patch(facecolor=(250/255, 250/255, 250/255), edgecolor="indianred", label="250")
        ]
        plt.legend(handles=ints_vis, loc="upper center", title="Intensity Comparisons", prop={"size": 15})
        plt.title("Average Intensity Histogram for MNIST " + self.ds_type)

        
        fig.add_subplot(1, 2, 2)
        ax = sns.kdeplot(data=otsu_threshs, fill=False, color="white", alpha=0)
        kdeL = ax.lines[0]
        xs = kdeL.get_xdata()
        ys = kdeL.get_ydata()
        ax.vlines(mean, 0, np.interp(mean, xs, ys), color="indianred", ls="--", label=f"Mean: {mean:.2f}")
        ax.fill_between(xs, 0, ys, where=(xs >= left_sig) & (xs <= right_sig), interpolate=True, color="indianred", alpha=0.4, label=f"Std.: {std:.2f}")
        ax.fill_between(xs, 0, ys, where=(xs < mean), color="skyblue", alpha=0.3, label=f"< Mean: {less_mean*100:.2f}%")
        ax.fill_between(xs, 0, ys, where=(xs >= mean), color="palegreen", alpha=0.3, label=f"≥ Mean: {high_mean*100:.2f}%")
        plt.title("KDE of Otsu Threshold Values on MNIST " + self.ds_type)
        plt.xlabel("Otsu Threshold Value")
        plt.legend()
        
        plt.show()    



    # Demonstrates the change in the token population (P) distribution as the threshold
    #   value changes
    def population_analysis(self, thresh_vals = [10, 60, 105, 130, 200], examples=10, start=0):

        # Containers for the token population distributions and the example images
        tk_pop_data = {
            "Threshold": [],
            "Token Population": [],
            "Means": []
        }
        eg_imgs = torch.empty((0, 28, 28))
        for i in range(examples):
            eg_imgs = torch.cat((eg_imgs, self.ds[start + i][0]), dim=0)

        # Collect the data
        for t_val in thresh_vals:
            counter = 0
            tk_pop = 0
            self.threshv = t_val / 255
            for i in range(len(self.ds)):
                img, _ = self.ds[i]
                tks = self.tokenize(img)
                tk_pop_data["Threshold"].append(t_val)
                tk_pop_data["Token Population"].append(tks.size(0))
                tk_pop += tks.size(0)
                if (i >= start) and (counter < examples):
                    eg_img, _ = self.visualise(tks)
                    eg_imgs = torch.cat((eg_imgs, eg_img.unsqueeze(0)), dim=0)
                    counter += 1
            tk_pop_data["Means"].append(tk_pop / len(self.ds))

        # Visualise the results
        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=tk_pop_data, x="Token Population", hue="Threshold", fill=True, alpha=0.3, linewidth=0, palette="crest")
        plt.vlines(tk_pop_data["Means"], 0, 1e-4, color="black", ls="-", alpha=0.5)
        plt.title("KDE of Token Population Under Varying Thresholds (" + self.ds_type + ")")
        plt.show()

        plt.figure(figsize=(12, 20))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(rearrange(eg_imgs[:examples], "(rw eg) h w -> (rw h) (eg w) ", rw=1), cmap="gray")

        plt.figure(figsize=(12, 20))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(rearrange(eg_imgs[examples:], "(rw eg) h w -> (rw h) (eg w) ", rw=len(thresh_vals)), cmap="gray")


