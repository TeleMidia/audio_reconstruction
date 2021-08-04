import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os
import glob
from PIL import Image
from tqdm import tqdm
import lib.jpeg as jpg
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse

exp_chart_folder = None
model_weights_folder1 = None
model_weights_folder2 = None
dict_chart_data = None
CONST_GAMA = 0.001
LAST_EPOCH = -1
BEST_VALIDATION_EPOCH = 0

class CustomMetric:
    def __init__(self):
        self.buffer_psnr = []
        self.buffer_nrmse = []
       
    def feed(self, batch_y, predictions):
        batch_size = predictions.shape[0]
        for index in range(0, batch_size):
            batch_y_r = batch_y[index,:,:,0]
            predictions_r = predictions[index,:,:,0]
            self.buffer_psnr = np.concatenate((self.buffer_psnr, peak_signal_noise_ratio(batch_y_r, predictions_r, data_range=1)), axis=None)
            self.buffer_nrmse = np.concatenate((self.buffer_nrmse, normalized_root_mse(batch_y_r, predictions_r)), axis=None)
            
    def result(self):
        return np.mean(self.buffer_psnr[~np.isinf(self.buffer_psnr)]), np.mean(self.buffer_nrmse)

    def reset_states(self):
        self.buffer_psnr = []
        self.buffer_nrmse = []


def check_experiment_folders():
    global exp_chart_folder, model_weights_folder1,  model_weights_folder2
    if exp_chart_folder is None or model_weights_folder1 is None or model_weights_folder2 is None:
        return False
    return True

def create_experiment_folders(exp_id):
    global exp_chart_folder, model_weights_folder1, model_weights_folder2
    exp_chart_folder = os.path.join("model_save", exp_id, "chart_data")
    model_weights_folder1 = os.path.join("model_save", exp_id, "model_last_epoch")
    model_weights_folder2 = os.path.join("model_save", exp_id, "model_best_valid")
    if not os.path.exists(exp_chart_folder):
        os.makedirs(exp_chart_folder)
    if not os.path.exists(model_weights_folder1):
        os.makedirs(model_weights_folder1)
    if not os.path.exists(model_weights_folder2):
        os.makedirs(model_weights_folder2)    
    return 

def get_exp_folder_last_epoch():
    return os.path.join(model_weights_folder1, "model")
    
def get_exp_folder_best_valid():
    return os.path.join(model_weights_folder2, "model")

def load_experiment_data():
    assert check_experiment_folders()
    global exp_chart_folder, dict_chart_data, LAST_EPOCH
    path =  os.path.join(exp_chart_folder, "data.txt")
    if os.path.exists(path):
        with open(path, "r") as file:
            dict_chart_data = eval(file.readline())
            #print(dict_chart_data)
            #print(dict_chart_data["epoch"])
            if len(dict_chart_data["epoch"]) > 0:
                LAST_EPOCH = int(dict_chart_data["epoch"][-1])
                #print(LAST_EPOCH)    
    else:
        dict_chart_data = {}
        dict_chart_data["epoch"] = []
        dict_chart_data["Train_MSE"] = []
        dict_chart_data["Valid_MSE_1"] = []
        dict_chart_data["Valid_MSE_2"] = []
        dict_chart_data["Valid_MSE_3"] = []
        dict_chart_data["PSNR_1"] = []
        dict_chart_data["PSNR_2"] = []
        dict_chart_data["PSNR_3"] = []
        dict_chart_data["NRMSE_1"] = []
        dict_chart_data["NRMSE_2"] = []
        dict_chart_data["NRMSE_3"] = []
        dict_chart_data["Best_Validation_Result"] = 0
        dict_chart_data["Best_Validation_Epoch"] = 0
    return

def get_model_last_data(mode="LastEpoch"):
    global LAST_EPOCH
    if mode =="LastEpoch":
        return LAST_EPOCH+1, dict_chart_data["Best_Validation_Result"]
    else: 
        return dict_chart_data["Best_Validation_Epoch"], dict_chart_data["Best_Validation_Result"]


def update_chart_data(epoch, train_mse, valid_mse, psnr, nrmse):
    assert check_experiment_folders()
    global exp_chart_folder,dict_chart_data
    assert dict_chart_data is not None
    path =  os.path.join(exp_chart_folder, "data.txt")

    if psnr[0] > dict_chart_data["Best_Validation_Result"]:
        dict_chart_data["Best_Validation_Result"] = psnr[0]
        dict_chart_data["Best_Validation_Epoch"] = epoch   

    dict_chart_data["epoch"].append(epoch)
    dict_chart_data["Train_MSE"].append(train_mse)
    dict_chart_data["Valid_MSE_1"].append(valid_mse[0])
    dict_chart_data["Valid_MSE_2"].append(valid_mse[1])
    dict_chart_data["Valid_MSE_3"].append(valid_mse[2])
    dict_chart_data["PSNR_1"].append(psnr[0])
    dict_chart_data["PSNR_2"].append(psnr[1])
    dict_chart_data["PSNR_3"].append(psnr[2])
    dict_chart_data["NRMSE_1"].append(nrmse[0])
    dict_chart_data["NRMSE_2"].append(nrmse[1])
    dict_chart_data["NRMSE_3"].append(nrmse[2])

    if os.path.exists(path):
        os.remove(path) 
    with open(path, "w") as file:
        file.write(str(dict_chart_data))
        
    return 

def annot_max(ax, x,y, op="min"):

    if op=="min":
        xmax = x[np.argmin(y)]
        ymax = y.min()
    else:
        xmax = x[np.argmax(y)]
        ymax = y.max()

    text= "epoch={}, result={:.6f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    arrowprops=dict(arrowstyle="->")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


def get_experiment_results():
    return { 
            "Best_Valid": dict_chart_data["Best_Validation_Result"],
            "Best_Epoch": dict_chart_data["Best_Validation_Epoch"],
            "PSNR_1": max(dict_chart_data["PSNR_1"]),
            "PSNR_2": max(dict_chart_data["PSNR_2"]),
             "PSNR_3": max(dict_chart_data["PSNR_3"]),
             "NRMSE_1": min(dict_chart_data["NRMSE_1"]),
             "NRMSE_2": min(dict_chart_data["NRMSE_2"]),
             "NRMSE_3": min(dict_chart_data["NRMSE_3"])
            }    

def draw_chart():
    global dict_chart_data

    if len(dict_chart_data["epoch"]) == 0:
        return

    fig, axs = plt.subplots(3, figsize=(15,15))
  
    axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Train_MSE"], linewidth=2, color="orange", label="Train_MSE")
    axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Valid_MSE_1"], linewidth=2, color="blue", label="Valid_MSE_1")
#     axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Valid_MSE_2"], linewidth=2, color="green", label="Valid_MSE_2")
#     axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Valid_MSE_3"], linewidth=2, color="red", label="Valid_MSE_3")
    axs[0].legend(frameon=False, loc='upper center', ncol=2)
    #annot_max(axs[0], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["Valid_MSE"]) )

    axs[1].plot(dict_chart_data["epoch"], dict_chart_data["PSNR_1"], linewidth=2, color="blue", label="PSNR_1")
#     axs[1].plot(dict_chart_data["epoch"], dict_chart_data["PSNR_2"], linewidth=2, color="green", label="PSNR_2")
#     axs[1].plot(dict_chart_data["epoch"], dict_chart_data["PSNR_3"], linewidth=2, color="red", label="PSNR_3")
    axs[1].legend(frameon=False, loc='upper center', ncol=1)
    #annot_max(axs[1], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["PSNR_1"]), op="max")

    axs[2].plot(dict_chart_data["epoch"], dict_chart_data["NRMSE_1"], linewidth=2, color="blue", label="NRMSE_1")
#     axs[2].plot(dict_chart_data["epoch"], dict_chart_data["NRMSE_2"], linewidth=2, color="green", label="NRMSE_2")
#     axs[2].plot(dict_chart_data["epoch"], dict_chart_data["NRMSE_3"], linewidth=2, color="red", label="NRMSE_3")
    axs[2].legend(frameon=False, loc='upper center', ncol=1)
    #annot_max(axs[4], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["NRMSE_1"]))

    plt.show()

   

def load_dataset(root_folder, replace_vec, load_gen=True, DCTScale=256, limit=None):
    
    IMG_SIZE = 200
    
    dataset_x_seismic = []
    dataset_x_dct = []
    dataset_y_seismic = []
    dataset_y_dct = []

    counter = 0

    qtable_luma_100, qtable_chroma_100 = jpg.generate_qtables(quality_factor=100)

    reg = "/*/*/*.tiff"
        
    for file_ in tqdm(glob.iglob(root_folder+reg, recursive=False)):
        file_path_x = file_.replace("\\", "/")
        file_path_y = file_path_x.replace(replace_vec[0], replace_vec[1])
        
        if load_gen:
            ext = file_path_y.split("/")[-1].split(".tiff")[0][-1]  
            file_path_y = file_path_y.replace("_"+ext+".tiff",".tiff")
        
       
        x_img = np.expand_dims(np.array(Image.open(file_path_x)), axis=2)
        assert x_img.shape == (IMG_SIZE, IMG_SIZE, 1)
        x_dct = None
        x_dct_path = file_path_x.replace(".tiff", "_dct_q100.npy")
        if os.path.exists(x_dct_path):
            x_dct = np.load(x_dct_path)
        else:
            x_dct = jpg.encode_image(x_img*DCTScale, qtable_luma_100, qtable_chroma_100)
            np.save(x_dct_path, x_dct)


        y_img = np.expand_dims(np.array(Image.open(file_path_y)), axis=2)
        assert y_img.shape == (IMG_SIZE, IMG_SIZE, 1) 
        y_dct = None
        y_dct_path = file_path_y.replace(".tiff", "_dct_q100.npy")
        if os.path.exists(y_dct_path):
            y_dct = np.load(y_dct_path)
        else:
            y_dct = jpg.encode_image(y_img*DCTScale, qtable_luma_100, qtable_chroma_100)
            np.save(y_dct_path, y_dct)
        
        
        dataset_x_seismic.append(x_img)
        dataset_y_seismic.append(y_img)

        dataset_x_dct.append(x_dct)
        dataset_y_dct.append(y_dct)

        counter += 1
        if limit != None and counter >= limit:
            break
        
    return np.array(dataset_x_seismic), np.array(dataset_y_seismic), np.array(dataset_x_dct), np.array(dataset_y_dct)

def load_dataset_from_step1(root_folder):
    IMG_SIZE = 200
    dataset_x_seismic = []
    dataset_y_seismic = []
    
    reg = "/*_x.npy"
    for file_ in tqdm(glob.iglob(root_folder+reg, recursive=False)):
        file_path_x = file_.replace("\\","/")
        file_path_y = file_path_x.replace("_x.npy", "_y.npy")
        
        x_img = np.load(file_path_x)
        dataset_x_seismic.append(x_img)
        
        y_img = np.load(file_path_y)
        dataset_y_seismic.append(y_img)
    
    return np.array(dataset_x_seismic), np.array(dataset_y_seismic), None, None    

def load_dataset_from_file(file_path, useDCT=False, DCTScale=256):
    
    IMG_SIZE = 200
    dataset_x_seismic = []
    dataset_x_dct = []
    dataset_y_seismic = []
    dataset_y_dct = []

    qtable_luma_100, qtable_chroma_100 = jpg.generate_qtables(quality_factor=100)

    f_ = open(file_path, "r")
    lines = f_.readlines()
    for line in tqdm(lines):
        line = line.replace("\n", "")
        data = line.split(";")
        file_path_x = data[0]
        file_path_x = file_path_x.replace("\\", "/")
        file_path_y = data[1]
        file_path_y = file_path_y.replace("\\", "/")
        
        x_img = np.expand_dims(np.array(Image.open(file_path_x)), axis=2)
        assert x_img.shape == (IMG_SIZE, IMG_SIZE, 1)
        if useDCT:
            x_dct = None
            x_dct_path = file_path_x.replace(".tiff", "_dct_q100.npy")
            if os.path.exists(x_dct_path):
                x_dct = np.load(x_dct_path)
            else:
                x_dct = jpg.encode_image(x_img*DCTScale, qtable_luma_100, qtable_chroma_100)
                np.save(x_dct_path, x_dct)
            
            dataset_x_dct.append(x_dct)
            
        y_img = np.expand_dims(np.array(Image.open(file_path_y)), axis=2)
        assert y_img.shape == (IMG_SIZE, IMG_SIZE, 1) 
        if useDCT:
            y_dct = None
            y_dct_path = file_path_y.replace(".tiff", "_dct_q100.npy")
            if os.path.exists(y_dct_path):
                y_dct = np.load(y_dct_path)
            else:
                y_dct = jpg.encode_image(y_img*DCTScale, qtable_luma_100, qtable_chroma_100)
                np.save(y_dct_path, y_dct)

            dataset_y_dct.append(y_dct)
            
        dataset_x_seismic.append(x_img)
        dataset_y_seismic.append(y_img)

        
    if useDCT:   
        return np.array(dataset_x_seismic), np.array(dataset_y_seismic), np.array(dataset_x_dct), np.array(dataset_y_dct)
    else:
        return np.array(dataset_x_seismic), np.array(dataset_y_seismic)

def random_mini_batches(X1, Y1, X2, Y2, mini_batch_size = 64, seed = 0):

    m = X1.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation]
    shuffled_Y1 = Y1[permutation]
    if X2 is not None:
        shuffled_X2 = X2[permutation]
        shuffled_Y2 = Y2[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y1 = shuffled_Y1[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = None
        if X2 is not None:
            mini_batch_X2 = shuffled_X2[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y2 = shuffled_Y2[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X1, mini_batch_Y1, mini_batch_X2, mini_batch_Y2)
        else:
            mini_batch = (mini_batch_X1, mini_batch_Y1, None, None)

        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X1 = shuffled_X1[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y1 = shuffled_Y1[num_complete_minibatches * mini_batch_size : m]

        mini_batch = None
        if X2 is not None:
            mini_batch_X2 = shuffled_X2[num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y2 = shuffled_Y2[num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X1, mini_batch_Y1, mini_batch_X2, mini_batch_Y2)
        else:
            mini_batch = (mini_batch_X1, mini_batch_Y1, None, None)

        mini_batches.append(mini_batch)
    
    return mini_batches


def get_patches_from_folder(folder):
    IMG_SIZE = 200
    patches = []
    qtd_images = 0
    files = glob.iglob(folder+"/*.tiff", recursive=False)
    
    for file in files:
            qtd_images+= 1
    
    for index in tqdm(range(0, qtd_images)):
    
        img = np.expand_dims(np.array(Image.open( folder+"/"+str(index)+".tiff" )), axis=2) 
        assert img.shape == (IMG_SIZE, IMG_SIZE, 1)
        patches.append(img)
        
    return np.array(patches)

def compose_seismogram(patches, per_column):
    column = None
    counter = 0
    final_seismogram = None
    qtd_patches = patches.shape[0]
    for index in range(0,qtd_patches):
        if counter < per_column:
            if column is None:
                column = patches[index,:,:,0]
            else:
                column = np.vstack((column, patches[index,:,:,0]))
            counter+= 1   
            if index == (qtd_patches-1):
                final_seismogram = np.hstack((final_seismogram, column))     
        else:
            if final_seismogram is None:
                final_seismogram = column
            else:
                final_seismogram = np.hstack((final_seismogram, column))
        
            column = patches[index,:,:,0]
            counter = 1
    return final_seismogram


def convert_batch_dct2seismo(batch, DCTScale=256):
    qtable_luma_100, qtable_chroma_100 = jpg.generate_qtables(quality_factor=100)
    quant = batch.shape[0]
    list_sample = []
    for index in range(quant):
            list_sample.append(jpg.decode_image(batch[index].copy(), qtable_luma_100, qtable_chroma_100))

    return np.array(list_sample)/DCTScale


def get_shift_scale_maxmin(train_x, train_y, valid_x, valid_y):
    
    SHIFT_VALUE_X = 0
    SHIFT_VALUE_Y = 0
    SCALE_VALUE_X = 0
    SCALE_VALUE_Y = 0

    if np.amin(valid_x) < np.amin(train_x):
        SHIFT_VALUE_X = np.amin(valid_x)
    else:
        SHIFT_VALUE_X = np.amin(train_x)

    if np.amin(valid_y) < np.amin(train_y):
        SHIFT_VALUE_Y = np.amin(valid_y)
    else:
        SHIFT_VALUE_Y = np.amin(train_y)

    if np.amax(valid_x) > np.amax(train_x):
        SCALE_VALUE_X = np.amax(valid_x)
    else:
        SCALE_VALUE_X = np.amax(train_x)

    if np.amax(valid_y) > np.amax(train_y):
        SCALE_VALUE_Y = np.amax(valid_y)
    else:
        SCALE_VALUE_Y = np.amax(train_y)

    
    SHIFT_VALUE_X = SHIFT_VALUE_X*-1
    SHIFT_VALUE_Y = SHIFT_VALUE_Y*-1
    SCALE_VALUE_X += SHIFT_VALUE_X
    SCALE_VALUE_Y += SHIFT_VALUE_Y

    return SHIFT_VALUE_X, SHIFT_VALUE_Y, SCALE_VALUE_X, SCALE_VALUE_Y 

def shift_and_normalize(batch, shift_value, scale_value):
    return ((batch+shift_value)/scale_value)+CONST_GAMA

def inv_shift_and_normalize(batch, shift_value, scale_value):
    return ((batch-CONST_GAMA)*scale_value)-shift_value


def add_margin_zeros(data_x, size=8, chan=1):

    data_x_size = data_x.shape[0]

    dataset_x = []

    zeros_1 = np.zeros((data_x.shape[1], size, chan))
    zeros_2 = np.zeros((size, data_x.shape[2]+size, chan))

    for i_nd in range(0,data_x_size):   
        tensor_x = np.hstack([data_x[i_nd], zeros_1])
        tensor_x = np.vstack([tensor_x, zeros_2])
        dataset_x.append(tensor_x)

    return np.array(dataset_x)

def remove_margin_zeros(data_x, size=8):

    data_x_size = data_x.shape[0]

    height = data_x.shape[1]
    width = data_x.shape[2]
    dataset_x = []

    for i_nd in range(0,data_x_size):
        tensor_x = data_x[i_nd,:(height-size),:,:]
        tensor_x = tensor_x[:,:(width-size),:] 
        dataset_x.append(tensor_x)

    return np.array(dataset_x)     

def load_single_seismogram(noisy_path, replace_str):
    dict_patches = {}
    DATA_SIZE = 200
    reg = "/*.tiff"
    for file_ in glob.iglob(noisy_path+reg, recursive=False):
        file_ = file_.replace("\\","/")
        key_ = int(os.path.basename(file_).replace(".tiff",""))
        
        dict_patches[key_] = file_
        
    dict_patches = dict_patches.items()
    dict_patches = sorted(dict_patches)
    #print(dict_patches)
    
    data_seismic_x = []
    data_seismic_y = []
    
    for file_ in dict_patches:
        key, file_ = file_
        x_data = np.expand_dims(np.array(Image.open(file_)), axis=2)
        assert x_data.shape == (DATA_SIZE, DATA_SIZE, 1)
        file_ = file_.replace(replace_str[0], replace_str[1])
        y_data = np.expand_dims(np.array(Image.open(file_)), axis=2)
        assert y_data.shape == (DATA_SIZE, DATA_SIZE, 1)
        
        data_seismic_x.append(x_data)
        data_seismic_y.append(y_data)
        
    return np.array(data_seismic_x), np.array(data_seismic_y)


dict_final_image = {}

def compose_final_image(key, data, pat_per_col, index, max_):
    global dict_final_image
    if not key in dict_final_image:
        dict_final_image[key] = {}
        dict_final_image[key]["col"] = None
        dict_final_image[key]["conter"] = 0
        dict_final_image[key]["image"] = None
    
    #print(dict_final_image[key]["conter"], "add to stack!")
    if dict_final_image[key]["col"] is None:
        dict_final_image[key]["col"] = data
    else:
        dict_final_image[key]["col"] = np.vstack((dict_final_image[key]["col"], data))

        
    if dict_final_image[key]["conter"] == pat_per_col or index == max_:
        #print(dict_final_image[key]["conter"],"next column!")
        
        if dict_final_image[key]["image"] is None:
            dict_final_image[key]["image"] = dict_final_image[key]["col"]
        else:
            dict_final_image[key]["image"] = np.hstack((dict_final_image[key]["image"], dict_final_image[key]["col"]))
        
        dict_final_image[key]["col"] = None   
        dict_final_image[key]["conter"] = 0 
    else: 
        dict_final_image[key]["conter"] =  dict_final_image[key]["conter"] + 1

def export_image_data(key):
    ret = dict_final_image[key]["image"]
    dict_final_image[key]["col"] = None
    dict_final_image[key]["conter"] = 0
    dict_final_image[key]["image"] = None

    return ret

def draw_trace(seismogram_x, seismogram_y, seismogram_p, trace_index):

    if trace_index < 0 or trace_index > seismogram_x.shape[0]:
        return None
    
    array_x  = seismogram_x[:,trace_index]
    array_y  = seismogram_y[:,trace_index]
    array_p  = seismogram_p[:,trace_index]

    t = np.arange(array_x.shape[0])

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, array_x, label='X')
    ax0.plot(t, array_y, label='Y')
    ax0.plot(t, array_p, label='P')
    ax0.set_xlabel("time")
    ax0.legend()