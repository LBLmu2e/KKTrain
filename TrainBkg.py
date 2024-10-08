#!/usr/bin/env python
# coding: utf-8

# # Hit classification

# Set initial variables for the "suffix" identifier for the data set and "filePath" the relative path to the data files. Remember to add "/" to the end of your path, for example "../../TrkAna/43291981/"

# In[ ]:


# import os

# suffix = "TTtpr"
# treename = "TAtpr"
# file_list = "/global/cfs/cdirs/m3712/Mu2e/TrkAna/60358177/files.txt"
# print("Using files in " + file_list)
print("Loaded TrainBkg")


# In this notebook we use Keras and XGBoost to distinguish between hits from conversion electrons and hits from other particles

# In[ ]:


import os
import ROOT
if not os.path.isfile(file_list):
    raise SystemExit("File list " +file_list + " doesn't exist! Exiting")
import uproot 
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, ReLU
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier


# In[ ]:


# rm -rf ./logs/


# In our problem we have a different number of signal and background entries in our input dataset. There are several techniques avaialable for _unbalanced_ datasets. Here we are using the most naive one, which is just using $\min(N_{sig}, N_{bkg})$ events. Then, we divide our input into the _training_, _validation_, and _test_ datasets.  Note that the datasets must be pruned to the nearest multiple of the batch size, otherwise the gradient calculation fails.

# In[ ]:


input_dataset = np.empty
temp = np.empty
signal = np.empty
backgnd = np.empty
##filter_name = "/dem|demtsh|demtshmc|demmc/i"
DownstreamElectron = "dem"
DownstreamElectronTrackStrawHit = "demtsh"
DownstreamElectronTrackStrawHitMC = "demtshmc"
DownstreamElectronMCTruth = "demmc"
thing = "/" + DownstreamElectron + "|" + DownstreamElectronTrackStrawHit + "|" + DownstreamElectronTrackStrawHitMC + "|" + DownstreamElectronMCTruth + "/i"
print(thing)

files = open(file_list, 'r')
for filename in files:
    print("Processing file " + filename)
    with uproot.open(filename) as file:
        for tree in treename:
            trkana = file[tree]["trkana"].arrays(filter_name="/" + DownstreamElectron + "|" + DownstreamElectronTrackStrawHit + "|" + DownstreamElectronTrackStrawHitMC + "|" + DownstreamElectronMCTruth + "/i") ##treename needs to be a list of strings
            trkana = trkana[(trkana['dem.goodfit']==1)&(trkana['dem.status']>0)&(trkana['demmc.proc']==167)]
            hstate = ak.concatenate(trkana['demtsh.state']).np.”()
            udoca = ak.concatenate(trkana['demtsh.udoca']).to_numpy()
            udoca = np.absolute(udoca)
            cdrift = ak.concatenate(trkana['demtsh.cdrift']).to_numpy()
            rdrift = ak.concatenate(trkana['demtsh.rdrift']).to_numpy()
            tottdrift = ak.concatenate(trkana['demtsh.tottdrift']).to_numpy()
            sint = ak.concatenate(trkana['demtsh.wdot']).to_numpy()
            sint = np.sqrt(1.0-sint*sint)
            plen = 6.25-rdrift*rdrift
            pmin = np.repeat(0.25,plen.shape[0])
            plen = np.sqrt(np.maximum(plen,pmin))
            udocasig = ak.concatenate(trkana['demtsh.udocavar']).to_numpy()
            udocasig = np.sqrt(udocasig)
            wdist = ak.concatenate(trkana['demtsh.wdist']).to_numpy()
            uupos = ak.concatenate(trkana['demtsh.uupos']).to_numpy()
            du = wdist-uupos
            du = np.absolute(du)
            rho = np.square(ak.concatenate(trkana['demtsh.poca.fCoordinates.fX']).to_numpy())
            rho = np.add(rho,np.square(ak.concatenate(trkana['demtsh.poca.fCoordinates.fY']).to_numpy()))
            rho = np.sqrt(rho)
            print("Processed file " + filename + " with %s hits"%hstate.shape[0])
            temp = np.vstack((udoca, cdrift, udocasig, tottdrift, du, rho)).T
            if input_dataset is np.empty:
                input_dataset = temp
            else:
                input_dataset = np.concatenate((input_dataset, temp))
            mcrel = []
            for i, this_dt in enumerate(trkana['demtsh.state']):
                mcrel.extend(trkana['demtshmc.rel._rel'][i][:len(this_dt)])
            mcrel = np.array(mcrel)
            rand = np.random.random_sample([mcrel.shape[0]])
            sig = (hstate>=-2) & (rand<0.05) & (mcrel==0) & (udoca < 50.0) & (du < 3000.0)
            bkg = (hstate>=-2) & (mcrel==-1) & (udoca < 50.0) & ( du < 3000.0)
            if signal is np.empty:
                signal = sig
                backgnd = bkg
            else:
                signal = np.concatenate((signal,sig))
                backgnd = np.concatenate((backgnd,bkg))
nhits=len(input_dataset)
nsignal=signal.sum()
nbackgnd=backgnd.sum()
print("Total dataset %s hits, %s signal and %s background"%(nhits,nsignal,nbackgnd))


# In[ ]:


min_len = min(len(input_dataset[signal]), len(input_dataset[backgnd]))
print("Number of events we train on for %s dataset after rebalancing is %s"%(suffix,min_len))
############################################PROBELM##########################################
bsize=32
# I need to double the batch_size when truncating as we divide the sample in half later for training
tsize=2*bsize
min_len = min_len - min_len%tsize
print("Training on %s matched hits"%min_len)
signal_dataset = input_dataset[signal][:min_len]
bkg_dataset = input_dataset[backgnd][:min_len]

balanced_input = np.concatenate((signal_dataset, bkg_dataset))
y_balanced_input = np.concatenate((np.ones(signal_dataset.shape[0]), np.zeros(bkg_dataset.shape[0])))

n_variables = balanced_input.shape[1]

x_ce_train, x_ce_test, y_ce_train, y_ce_test = train_test_split(balanced_input, y_balanced_input, test_size=0.5, random_state=42)
x_ce_test, x_ce_valid, y_ce_test, y_ce_valid = train_test_split(x_ce_test, y_ce_test, test_size=0.5, random_state=42)


# In[ ]:


udoca_sig = []
tottdrift_sig = []
cdrift_sig = []
udocasig_sig = []
du_sig = []
rho_sig = []

for i in range(signal_dataset.shape[0]):
    udoca_sig.append(signal_dataset[i][0])
    
for i in range(signal_dataset.shape[0]):
    cdrift_sig.append(signal_dataset[i][1])
    
for i in range(signal_dataset.shape[0]):
    udocasig_sig.append(signal_dataset[i][2])

for i in range(signal_dataset.shape[0]):
    tottdrift_sig.append(signal_dataset[i][3])

for i in range(signal_dataset.shape[0]):
    du_sig.append(signal_dataset[i][4])

for i in range(signal_dataset.shape[0]):
    rho_sig.append(signal_dataset[i][5])


# In[ ]:


udoca_back = []
tottdrift_back = []
cdrift_back = []
udocasig_back = []
du_back = []
rho_back = []
    
for i in range(bkg_dataset.shape[0]):
    udoca_back.append(bkg_dataset[i][0])
    
for i in range(bkg_dataset.shape[0]):
    cdrift_back.append(bkg_dataset[i][1])
    
for i in range(bkg_dataset.shape[0]):
    udocasig_back.append(bkg_dataset[i][2])

for i in range(bkg_dataset.shape[0]):
    tottdrift_back.append(bkg_dataset[i][3])
    
for i in range(bkg_dataset.shape[0]):
    du_back.append(bkg_dataset[i][4])
    
for i in range(bkg_dataset.shape[0]):
    rho_back.append(bkg_dataset[i][5])


# In[ ]:



label1 = "Unbiased DOCA"
# label1b = "Unbiased DOCA Background"
plt.hist(udoca_sig, label = "Unbiased DOCA Signal", bins=100, range=(0,15))
plt.hist(udoca_back,label= "Unbiased DOCA Background", histtype='step', bins=50, range=(-3.0,5.0))
plt.xlabel("mm")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/"+ "TrainBkg" + suffix + label1 + ".pdf")
plt.clf()


# In[ ]:

label2 = "Drift Radius"
# label2b = "Drift Radius Background"
plt.hist(cdrift_sig,label = "Drift Radius Signal", bins=50,range=(-3.0,5.0))
plt.hist(cdrift_back,label = "Drift Radius Background", histtype='step', bins=50,range=(-3.0,5.0))
plt.xlabel("mm")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/"+ "TrainBkg" + suffix + label2 + ".pdf")
plt.clf()


# In[ ]:

label3 = "DOCA Variance"
# label3b = "DOCA Variance Background"
plt.hist(udocasig_sig, label = "DOCA Variance Signal", bins=50,range=(0,5))
plt.hist(udocasig_back, label = "DOCA Variance Background", histtype='step', bins=50,range=(0,5))
plt.xlabel("mm^2")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/"+ "TrainBkg" + suffix + label3 + ".pdf") 
plt.clf()


# In[ ]:

label4 = "Time-Over-Threshold Drift Time"
# label4b = "Time-Over-Threshold Drift Time Background"
plt.hist(tottdrift_sig,label = "Time-Over-Threshold Drift Time Signal", bins=50, range=(0,40))
plt.hist(tottdrift_back,label = "Time-Over-Threshold Drift Time Background", histtype='step', bins=50, range=(0,40))
plt.xlabel("ns")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/" +"TrainBkg" +  suffix + label4 + ".pdf")
plt.clf()

# In[ ]:

label5 = "WDist - Unbiased U Position Difference"
# label5b = "WDist - Unbiased U Position Difference Background"
plt.hist(du_sig,label = "WDist - Unbiased U Position Difference Signal", bins=50, range=(0,800))
plt.hist(du_back,label = "WDist - Unbiased U Position Difference Background", histtype='step', bins=50, range=(0,800))
plt.xlabel("mm")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/" + "TrainBkg" + suffix + label5 + ".pdf")
plt.clf()


# In[ ]:

label6 = "Rho"
# label6b = "Rho Background"
plt.hist(rho_sig, label = "Rho Signal", bins=50, range=(350,700))
plt.hist(rho_back, label = "Rho Background", histtype='step', bins=50, range=(350,700))
plt.xlabel("mm")
plt.ylabel("Number of Hits")
plt.legend()
plt.savefig("plots/" + "TrainBkg" + suffix + label6 + ".pdf")
plt.clf()


# Then, we concatenate each hit variables into a single, large numpy array. These arrays are then stacked in a single bi-dimensional array with `np.vstack`, which will be our input dataset used for the training of the machine learning algorithms.

# Here we then assign a label to each hit as _signal_ or _background_, depending on the Monte Carlo truth information. Since the dimension of  `detshmc.rel._rel` is not guaranteed to be the same as the dimension of `detsh` we need to loop over all the entries.

# ## Create and train a multi-layer perceptron

# Here we create a _multi-layer perceptron_ (MLP) model which consists of 3 fully-connected (or _dense_) layers, each one followed by a _dropout_ layer, which helps to avoid overfitting. The model is trained using the [Adam](https://arxiv.org/abs/1412.6980) optimizer and trained for 50 epochs or until the validation loss doesn't improve for 5 epochs (`early_stop`). The model we save must be created first, with an explicit input layer with explicit batch_size, otherwise it can't be parsed by the TMVA::SOFIE parser we use to generate a C++ inference function downstream. 

# The output model must have batch_size=1, otherwise the SOFIE inference function will assume that many input variables at a time.  Training (gradient calculation) however is much more efficient with a larger batch size, so we construct a separate model for that.  After training and before saving, we'll copy the weights (which don't depend on batch_size) from the trained model to the output model.  This should be unnecessary in the next verison of ROOT.
# 
# We should initialize the model by reading a previous iteration. TODO

# In[ ]:


# load previous training if it exists
modelfile = "models/TrainBkg" + suffix + ".h5"
if os.path.isfile(modelfile):
    model = keras.models.load_model(modelfile)
    print("Loading model from file " +modelfile)
else:
    print("Creating model " +modelfile)
    lay0=Input(shape=(n_variables,),batch_size=bsize)
    lay1=Dense(2*n_variables, activation='relu')(lay0)
    lay2=Dense(2*n_variables, activation='relu')(lay1)
    lay3=Dense(2*n_variables, activation='relu')(lay2)
    lay4=Dense(1,activation='sigmoid')(lay3)
    model=Model(inputs=lay0,outputs=lay4)

opt = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy',metrics='accuracy',optimizer=opt)
early_stop = EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-5, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_ce_train, y_ce_train,
                          batch_size=bsize,
                          epochs=200,
                          verbose=1,
                          validation_data=(x_ce_valid, y_ce_valid),
                          callbacks=[early_stop, tensorboard_callback]
                         )


# In[ ]:


plt.plot(history.history['val_loss'],label="val loss")
plt.plot(history.history['loss'],label="loss")
plt.legend()
plt.savefig("training_plots/"+ suffix + "Validation_Loss_Plot.pdf")

# get_ipython().run_line_magic('load_ext', 'tensorboard')
# #rm -rf ./logs/
# get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# ## Create and train a Boosted Decision Tree
# Here, instead of using a MLP, we use a [_Gradient Boosted Decision Tree_](https://xgboost.readthedocs.io/en/stable/) (BDT) to distinguish between signal (true CE hits) and background (fake CE hits). We use the defualt hyperparameters.

# In[ ]:


model_xgboost = XGBClassifier(use_label_encoder=False)
model_xgboost.fit(x_ce_train, y_ce_train)


# Here we can finally apply our two models (the MLP and the BDT) to our test datasets and create the corresponding ROC curves.

# In[ ]:


#prediction_ce = model.predict(x_ce_test).ravel()
prediction_ce = model.predict(x_ce_test)
fpr_ce, tpr_ce, th_ce = roc_curve(y_ce_test,  prediction_ce)
auc_ce = roc_auc_score(y_ce_test, prediction_ce)


# In[ ]:


prediction_xgboost = model_xgboost.predict_proba(x_ce_test)[:,1]
fpr_xgboost, tpr_xgboost, th_xgboost = roc_curve(y_ce_test,  prediction_xgboost)
auc_xgboost = roc_auc_score(y_ce_test, prediction_xgboost)


# The plot of the ROC curves clearly shows that the BDT outperforms the MLP. In principle, however, it should be possible to improve the MLP performances by optimizing the hyperparameters (learning rate, hidden layers, activation functions, etc.).

# In[ ]:


fig, ax = plt.subplots(1,1)
ax.plot(tpr_ce,1-fpr_ce,label=f'MLP AUC: {auc_ce:.4f}')
# ax.plot(tpr_xgboost,1-fpr_xgboost,label=f'BDT AUC: {auc_xgboost:.4f}')

ax.legend()
ax.set_aspect("equal")
ax.set_xlabel("Signal efficiency")
ax.set_ylabel("Background rejection")
ax.set_xlim(0.8,1.05)
ax.set_ylim(0.8,1.05)

fig.savefig("plots/TrainBkg" + suffix + ".pdf")



# Now we save our model in HDF5 format.  This can be used as input to the SOFIE parser.  Note that the kernel must be restarted when saving this file, as re-running individual cells increments the layer numbers in the hdf5 file, causing the SOFIE parser to fail.  This causes the spurious tensorflow warning about the model not having been built.

# In[21]:



model.summary()
model.save("models/TrainBkg" + suffix + ".h5")



# Now we save our model in HDF5 format.  This can be used as input to the SOFIE parser.  Note that the kernel must be restarted when saving this file, as re-running individual cells increments the layer numbers in the hdf5 file, causing the SOFIE parser to fail.  This causes the spurious tensorflow warning about the model not having been built.
# modelFile = "models/TrainBkg" + suffix + ".h5"
# codename = "code/TrainBkg" + suffix + ".hxx";
# output_model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)
# output_model.Generate()
# output_model.OutputGenerated(codename)
with open("SOFIEConvert.py") as f:
    exec(f.read())
