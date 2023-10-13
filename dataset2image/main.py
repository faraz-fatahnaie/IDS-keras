import pandas as pd
import numpy as np

from dataset2image.Cart2Pixel import Cart2Pixel
from dataset2image.ConvPixel import ConvPixel
from utils import parse_data
from pathlib import Path


def deepinsight(param, config):
    # param = {"Max_A_Size": 14, "Max_B_Size": 14, "Dynamic_Size": False, 'Method': 'tSNE',
    #          "seed": 1401,
    #          "dir": '/home/faraz/PycharmProjects/IDS/dataset2image', "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
    #          "LoadFromPickle": False, "mutual_info": True,  # Mean or MI
    #          "hyper_opt_evals": 50, "epoch": 2, "No_0_MI": False,  # True -> Removing 0 MI Features
    #          "autoencoder": False, "cut": None, "enhanced_dataset": "gan"  # gan, smote, adasyn, ""None""
    #          }

    # files name
    name = "_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))
    if param["No_0_MI"]:
        name = name + "_No_0_MI"
    if param["mutual_info"]:
        name = name + "_MI"
    else:
        name = name + "_Mean"
    # if image_model["custom_cut"] is not None:
    #     name = name + "_Cut" + str(image_model["custom_cut"])
    filename_train = "train" + name + ".npy"
    filename_test = "test" + name + ".npy"

    try:
        XGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_train)))
        XTestGlobal = np.load(str(Path(config['DATASET_PATH']).joinpath(filename_test)))
        print(f"Train and Test Images Loaded with the Size of {np.shape(XGlobal)} and {np.shape(XTestGlobal)},"
              f" respectively.")
        return np.array(XGlobal), np.array(XTestGlobal)

    except:
        # train_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('train_' + config['CLASSIFICATION_MODE'] + '.csv'))
        train_df = pd.read_csv(
            'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\KDD_CUP99\\train_binary_2neuron_labelOnehot.csv')

        x_train, y_train = parse_data(train_df, dataset_name=config['DATASET_NAME'], mode='df',
                                      classification_mode=config['CLASSIFICATION_MODE'])
        print(f'train shape: x=>{x_train.shape}, y=>{y_train.shape}')
        y_train = y_train.to_numpy()

        # test_df = pd.read_csv(Path(config['DATASET_PATH']).joinpath('test_' + config['CLASSIFICATION_MODE'] + '.csv'))
        test_df = pd.read_csv(
            'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\KDD_CUP99\\test_binary_2neuron_labelOnehot.csv')
        x_test, y_test = parse_data(test_df, dataset_name=config['DATASET_NAME'], mode='df',
                                    classification_mode=config['CLASSIFICATION_MODE'])
        print(f'test shape: x=>{x_test.shape}, y=>{y_test.shape}')

        np.random.seed(param["seed"])
        print("transposing")
        # q["data"] is matrix T in paper (transpose of dataset without labels)
        # max_A_size, max_B_size is n and m in paper (the final size of generated image)
        # q["y"] is labels
        q = {"data": np.array(x_train.values).transpose(), "method": param["Method"],
             "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": y_train.argmax(axis=-1)}
        print(q["method"])
        print(q["max_A_size"])
        print(q["max_B_size"])

        # generate images
        XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                    mutual_info=param["mutual_info"], params=param, only_model=False)

        np.save(str(Path(config['DATASET_PATH']).joinpath(filename_train)), XGlobal)
        print("Train Images generated and train images with labels are saved with the size of:", np.shape(XGlobal))

        # generate testing set image
        if param["mutual_info"]:
            x_test = x_test.drop(x_test.columns[toDelete], axis=1)

        x_test = np.array(x_test).transpose()
        print("generating Test Images for X_test with size ", x_test.shape)

        if image_model["custom_cut"] is not None:
            XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                     image_model["A"], image_model["B"], custom_cut=range(0, image_model["custom_cut"]))
                           for i in range(0, x_test.shape[1])]
        else:
            XTestGlobal = [ConvPixel(x_test[:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                     image_model["A"], image_model["B"])
                           for i in range(0, x_test.shape[1])]

        np.save(str(Path(config['DATASET_PATH']).joinpath(filename_test)), XTestGlobal)
        print("Test Images generated and test images with labels are saved with the size of:", np.shape(XTestGlobal))

        return np.array(XGlobal), np.array(XTestGlobal)

# if __name__ == '__main__':
#     deepinsight()
