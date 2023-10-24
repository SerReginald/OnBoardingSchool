import glob
import numpy as np
import imageio.v3 as iio
from math import sqrt
import matplotlib.pyplot as plt

def lecture_shape_1channel(dir, extension, lst_shape):

  # Read the files with extension "ext" in directory "dir"
  # Initialize the input data_train (input images) and output_train (expected class)
  # The expected class is inferred from the file name

  files = glob.glob(dir+extension)
  data_train = []
  output_train = []
  for file in files:
    image = np.array(iio.imread(file))[:,:,1].reshape((64,64,1))
    data_train.append(image)
    for ind, shape in enumerate(lst_shape):
      if shape in file:
        output_train.append(ind)

  return np.array(data_train), np.array(output_train)

def lecture_shape_3channels(dir, extension, lst_shape):

  # Read the files with extension "ext" in directory "dir"
  # Initialize the input data_train (input images) and output_train (expected class)
  # The expected class is inferred from the file name

  files = glob.glob(dir+extension)
  data_train = []
  output_train = []
  for file in files:
    image = np.array(iio.imread(file)).reshape((64,64,3))
    data_train.append(image)
    for ind, shape in enumerate(lst_shape):
      if shape in file:
        output_train.append(ind)

  return np.array(data_train), np.array(output_train)


def draw_history(history, **kwargs):
  
  plt.rcParams.update({'font.size': 10})
    # Plot the loss and accuracy curves for training and validation

  liste_cle = [cle for cle in history.history]
  liste_metrics = [cle for cle in liste_cle if not "val" in cle]

  print(liste_metrics, liste_cle)

  fig, ax = plt.subplots(int((len(liste_cle) + 1) / 2))

  couleur = ['b', 'm', 'r', 'c', 'y', 'g']

  for i, cle in enumerate(liste_cle):
    metrics = cle
    if cle.find('val') >= 0:
      metrics = cle[4:]
    ind_gr = liste_metrics.index(metrics)
    ax[ind_gr].plot(history.history[cle], color=couleur[i%6], label=cle)
    ax[ind_gr].legend()

  plt.show(**kwargs)
    

def draw_multiple_images(input_test, output_true, output_predict, lst_shape, nb_col, im_size, **kwargs):

  # Plot all the images ; the predicted class is written in red when it is wrong, in green otherwise

  plt.rcParams.update({'font.size': 96})

  nb_sample = np.shape(input_test)[0]
  nb_row = 1 + int((nb_sample - 1) / nb_col)
  print(nb_sample, nb_row)
  fig, ax = plt.subplots(nb_row, nb_col, figsize=im_size[0:2], constrained_layout=True)
  i = 0          

  for image, class_true, class_predict in zip(input_test, output_true, output_predict):
    image = np.reshape(image, im_size)
    cr = 'g' if np.argmax(class_true) == np.argmax(class_predict) else 'r'
    col = int(i % nb_col)
    row = int(i / nb_col)
    ax[row, col].imshow(image)
    ax[row, col].axis('off')
    ax[row, col].text(
      0.5, 0.1, lst_shape[np.argmax(class_predict)][:3]+'.',
      horizontalalignment='center',
      verticalalignment='center',
      transform=ax[row, col].transAxes,
      color=cr)
    i = i + 1

  plt.show(**kwargs)

