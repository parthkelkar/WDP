# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:33:14 2021

@author: kelka
"""
import matplotlib.pyplot as plt
import numpy as np
import iris
import iris.analysis.cartography
import iris.analysis.stats
import cf_units
# import mask_removal as  m

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

path1 = 'annual'
model='BCC_ESM1'
path='D:/project/Birmingham-gpp-data/'+path1+'/'    

filename1='CMIP6_BCC-ESM1_Amon_piControl_r1i1p1f1_pr_1850-2300.nc'
filename2='CMIP6_BCC-ESM1_Amon_piControl_r1i1p1f1_tas_1850-2300.nc'
filename3='CMIP6_BCC-ESM1_Lmon_piControl_r1i1p1f1_gpp_1850-2300.nc'


pr = iris.load_cube(path+filename1)
tas = iris.load_cube(path+filename2)
gpp = iris.load_cube(path+filename3)


tcoord = tas.coord('time')
tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')
tcoord = pr.coord('time')
tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')
tcoord = gpp.coord('time')
tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')


tas.convert_units('celsius')
pr.convert_units('kg m-2 days-1')
gpp.convert_units('kg m-2 days-1')


gpp.coord('latitude').points=pr.coord('latitude').points
gpp.coord('latitude').bounds=pr.coord('latitude').bounds
gpp.coord('longitude').points=pr.coord('longitude').points
gpp.coord('longitude').bounds=pr.coord('longitude').bounds



# mean_temp=tas.collapsed(['time','latitude','longitude'], iris.analysis.MEAN)

# data normalization range between 0-1
tas_min=tas.data.min()
tas_max=tas.data.max()

tas_normal=(tas-tas_min)/(tas_max-tas_min)


pr_min=pr.data.min()
pr_max=pr.data.max()
pr_normal=(pr-pr_min)/(pr_max-pr_min)



gpp_min=gpp.data.min()
gpp_max=gpp.data.max()
gpp_normal=(gpp-gpp_min)/(gpp_max-gpp_min)


#train test split of the data


time_constraint_train = iris.Constraint(year=lambda cell: 1860 < cell < 2000)
time_constraint_test = iris.Constraint(year=lambda cell: 2000 < cell < 2100)

train_tas=tas_normal.extract(time_constraint_train)
train_pr=pr_normal.extract(time_constraint_train)
train_gpp=gpp_normal.extract(time_constraint_train)


test_tas=tas_normal.extract(time_constraint_test)
test_pr=pr_normal.extract(time_constraint_test)
test_gpp=gpp_normal.extract(time_constraint_test)




#intial model

v=len(train_tas.coord('time').points)

input_pr = keras.Input(shape=(64, 128,1), name="pr")



input_tas=keras.Input(shape=(64, 128,1), name="tas")


x1=layers.Conv2D(64, (2,2), activation='tanh', padding="same",dilation_rate=2)(input_pr)
x1=layers.MaxPooling2D((2, 2),padding="valid")(x1)
x1=layers.Conv2D(128, (3, 3), activation='tanh',padding="same")(x1)
x1=layers.MaxPooling2D((2, 2),padding="valid")(x1)
x1=layers.Conv2D(256, (3, 3), activation='tanh',padding="same")(x1)
x1=layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x1)
x1=layers.Conv2D(128, (3, 3), activation='tanh',padding="same")(x1)
x1=layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x1)




x2=layers.Conv2D(64, (2,2), activation='tanh', padding="same",dilation_rate=2)(input_tas)
x2=layers.MaxPooling2D((2, 2),padding="valid")(x2)
x2=layers.Conv2D(128, (3, 3), activation='tanh',padding="same")(x2)
x2=layers.MaxPooling2D((2, 2),padding="valid")(x2)
x2=layers.Conv2D(256, (3, 3), activation='tanh',padding="same")(x2)
x2=layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x2)
x2=layers.Conv2D(128, (3, 3), activation='tanh',padding="same")(x2)
x2=layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x2)



x = layers.concatenate([x1, x2])
x=layers.Conv2D(64, (3, 3), activation='tanh',padding="same",dilation_rate=2)(x)
gpp_output=layers.Conv2D(1, (5, 5), activation='relu',padding="same",name='gpp_output',dilation_rate=1)(x)





model = keras.Model(inputs=[input_pr, input_tas], outputs=[gpp_output])
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=25,
        verbose=1,
    ),
    # keras.callbacks.TensorBoard(
    # log_dir="D:\project\Birmingham-gpp-data\plots\code\mymodel_1\logs",
    # histogram_freq=0,  # How often to log histogram visualizations
    # embeddings_freq=0,  # How often to log embedding visualizations
    # update_freq="epoch",
    # ) 
] 


model.fit(
    {"input_pr": train_pr.data.filled(0), "input_tas": train_tas.data.filled(0)},
    {"gpp_output": train_gpp.data.filled(0) },
    batch_size=32,
    epochs=200,
    callbacks=callbacks,
    validation_split=0.2
)

# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)





z=model.predict( {"input_pr": test_pr.data.filled(0), "input_tas": test_tas.data.filled(0)} )


















