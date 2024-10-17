import pandas as pd
import pickle
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import fnmatch
# import zipfile as zip
import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pred import load_data, load_data_for_keras
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from keras.layers import Input, Dense
from keras.models import Model
from keras import ops
from keras import layers
import tensorflow as tf
import multiprocessing

import pred

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def vae_loss(self,y_true, y_pred, z_mean, z_log_var): 
        """
        Loss function for the VAE model that computes the reconstruction loss and the KL divergence loss
        """
        def vae_reconstruction_loss(y_true, y_pred):
            reconstruction_ratio = 1000
            reconstruction_loss = tf.reduce_mean(ops.square(y_true-y_pred), axis=1)
            return reconstruction_loss*reconstruction_ratio
        def vae_kl_loss(z_mean, z_log_var):
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
            return kl_loss
        
        def vae_loss(y_true, y_pred, z_mean, z_log_var):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(z_mean, z_log_var)
            return (reconstruction_loss + kl_loss), reconstruction_loss, kl_loss
        
        return vae_loss(y_true, y_pred, z_mean, z_log_var)

    def train_step(self, data):
        with tf.GradientTape() as tape: 
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            # reconstruction_loss = keras.losses.binary_crossentropy(data[0], reconstruction)
            total_loss, reconstruction_loss, kl_loss = self.vae_loss(data[0], reconstruction, z_mean, z_log_var)

        gradient = tape.gradient(total_loss, self.trainable_weights) 

        self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data[0])
        reconstruction = self.decoder(z)
        # reconstruction_loss = keras.losses.binary_crossentropy(data[0], reconstruction)
        total_loss, reconstruction_loss, kl_loss = self.vae_loss(data[0], reconstruction, z_mean, z_log_var)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

class Autoencoder:
    def __init__(self, encoding_dim, input_shape, vae=False, intermediate_dim=None):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self.vae = vae
        if vae:
            self.model = self.build_vae(self.input_shape, latent_dim=encoding_dim, intermediate_dim=intermediate_dim)
        else: 
            self.model = self.autoencoder_build()

    def fit(self, tX, epochs=100, batch_size=125, shuffle=True, validation_split=0.2):
        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        # tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
        self.model.fit(tX, tX, epochs=epochs, batch_size=batch_size, 
                       shuffle=shuffle, validation_split=validation_split, callbacks=[reduce_lr, early_stopping], verbose=1)
    
    def save(self, path):
        if self.vae:
            self.model.encoder.save(path)
        else:
            self.model.save(path)


    def autoencoder_build(self) :

        # Fit and transform the data
        input_layer = Input(shape=(self.input_shape,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_shape, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        # print(autoencoder.summary())

        return autoencoder 

    def build_vae(self, input_shape, intermediate_dim=512, latent_dim=2):
        """
        Variational Autoencoder model
        """
        # Encoder
        inputs = Input(shape=(input_shape,), name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, activation='relu', name='z_mean')(x)
        z_log_var = Dense(latent_dim, activation='relu', name='z_log_var')(x)

        # Sample from the latent space
        z = Sampling(name='sampling')([z_mean, z_log_var]) 

        # Encoder
        encoder = Model(inputs,  [z_mean, z_log_var, z], name='encoder')
        # encoder.summary()

        # Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()

        # VAE
        vae = VAE(encoder, decoder)
        vae.compile(optimizer='adam')

        return vae


    def predict(self, valX):
        """
        Transforms the input into expected shape and passes it through the encoder

        Input shape: (samples, TSTEP, moneyness)
        Output shape: (samples, encoding_dim*tX.shape[-1])
        """
        if self.vae:
            encoder = self.model.encoder 
        else:
            encoder = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        return encoder.predict(valX)

def autoencoder_transform(encoder , tX, type='skew', vae=False):
    # Strip the moneyness or term structure feature
    if type == 'skew':
        structure = tX[:, -1:]
        tX = tX[:, :-1]
    else: # point model
        structure = tX[:, -2:]
        tX = tX[:, :-2]
    
    tX_transformed = encoder.predict(tX)

    # Because vae is outputing the z_mean, z_log_var and z
    if vae:
        tX_transformed = tX_transformed[2]

    # Add the structure back So we end up with our encoded features + structure at the end of each sample
    tX_transformed = np.concatenate([tX_transformed, structure], axis=1)

    return tX_transformed



def autoencoder_fit(tX, ty, encoding_dim, intermediate_dim,  vae=False):
    """
    Uses an autoencoder to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample

    Output: encoder, ridge model 
    """
    encoder = Autoencoder(encoding_dim, tX.shape[1], vae=vae, intermediate_dim=intermediate_dim)
    encoder.fit(tX)
    return encoder 

def pca_predict(valX, model, n_components, TSTEPS):
    valX_transform = pca_transform(valX, n_components, TSTEPS)
    return model.predict(valX_transform)

def pca_transform(tX, n_components=3):
    """
    uses PCA to extract features from the data

    Input shape: (samples, TSTEP, moneyness)

    Ouput shape: (samples, components*tX.shape[-1])
    """
    # Fit and transform the data
    pca = PCA(n_components=n_components)
    try:
        tX_transform = pca.fit_transform(tX)
        return tX_transform
    except:
        print("Error in PCA trying again")

    tX_transform = pca.fit_transform(tX)

    return tX_transform
    
def har_transform(tX, TSTEPS=21, type='skew'):
    """
    Transform the given input data to the HAR method of feature extraction
    Expects 2D input shape

    Input shape: (samples, TSTEP, moneyness)
    TSTEP must be of size 21 as a month of IV data is required for the HAR method

    Output shape: (samples, tx.shape[-1]*3)
    """
    if TSTEPS != 21:
        raise ValueError('TSTEP must be 21 for HAR method of feature extraction')

    # Get average skew for 21 days
    skew1 = np.mean(tX[:, :], axis=1, keepdims=True)
    skew2 = np.mean(tX[:, -5:], axis=1, keepdims=True)
    skew3 = np.mean(tX[:, -1:], axis=1, keepdims=True)
    tX = np.concatenate([skew1, skew2, skew3], axis=1)

    return tX 

def extract_features(tX, tY, model_name='pca', TSTEPS=21, type='skew', dd='./figs', otype='call', m=0, save=True):
    """
    Extracts features from the given data
    """
    if model_name == 'har':
        if type=='skew':
            # Reshape to 3D
            tX = tX.reshape(tX.shape[0], TSTEPS, tX.shape[1]//TSTEPS)

        tX = har_transform(tX, TSTEPS=TSTEPS, type=type)
    elif model_name == 'autoencoder':
        encoding_dim = TSTEPS//2 

        if type == 'skew':
            num_points = tX.shape[1]//TSTEPS
            encoding_dim = encoding_dim*num_points

        encoder = autoencoder_fit(tX, tY, encoding_dim=encoding_dim, intermediate_dim=None, TSTEPS=TSTEPS)

        # Transform data
        tX = encoder.predict(tX)

        # Save encoder model
        if save:
            if dd != './gfigs':
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, m, otype))
            else:
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, m, otype))

        return tX, encoder
        
    elif model_name == 'vae':
        encoding_dim = TSTEPS//2

        if type == 'skew':
            num_points = tX.shape[1]//TSTEPS
            latent_dim = encoding_dim*num_points
            intermediate_dim = (encoding_dim+1)*num_points
        else:
            latent_dim = encoding_dim
            intermediate_dim = (encoding_dim+1)

        vae_encoder = autoencoder_fit(tX, tY, encoding_dim=latent_dim, intermediate_dim=intermediate_dim, vae=True)

        # Transform data
        tX = vae_encoder.predict(tX)[2]

        # Save encoder model
        if save:
            if dd != './gfigs':
                vae_encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, m, otype))
            else:
                vae_encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, m, otype))
        
        return tX, vae_encoder

    else: # PCA
        n_components = (TSTEPS//2) * tX.shape[-1] 
        tX = pca_transform(tX, tY, n_components=n_components)

    return tX 

def validation(vX, vY, model):
    """
    Validate the model on the validation set
    """
    vYp = model.predict(vX)
    rmse = root_mean_squared_error(vY, vYp)
    mape = mean_absolute_percentage_error(vY, vYp)
    r2 = r2_score(vY, vYp)
    print('RMSE: ', rmse)
    print('MAPE: ', mape)
    print('R2: ', r2)
    return rmse, mape, r2

def tskew_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./tskew_feature_models'):
        os.makedirs('./tskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    # Moneyness range to iterate over
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    count = 0

    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1

        # XXX: shape = samples, TSTEPS, moneyness, term structure
        tskew = tX[:, :, j]
        tYY = tY[:, j]
        tskew = tskew.reshape(tskew.shape[0], tskew.shape[1]*tskew.shape[2])

        if model_name == 'har' and TSTEPS != 21:
            continue 

        # Extract features
        tskew = extract_features(tskew, tYY, model_name=model_name, TSTEPS=TSTEPS, type='skew', dd=dd, otype=otype, m=m)

        # XXX: Add m to the sample set
        ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
        tskew = np.append(tskew, ms, axis=1)

        # Fit Regression model
        model = Ridge()
        model.fit(tskew, tYY)

        # Validate the model
        validation(tskew, tYY, model)

        if dd != './gfigs':
            with open('./tskew_feature_models/%s_ts_%s_%s_%s.pkl' % (model_name, TSTEPS, m, otype), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./tskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' % (model_name, TSTEPS, m, otype), 'wb') as f:
                pickle.dump(model, f)


def mskew_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./mskew_feature_models'):
        os.makedirs('./mskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    # Term structure range to iterate over
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]

    count = 0

    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]
        mskew = mskew.reshape(mskew.shape[0], mskew.shape[1]*mskew.shape[2])

        mskew, encoder = extract_features(mskew, tYY, model_name=model_name, TSTEPS=TSTEPS, type='skew', dd=dd, otype=otype, m=t, save=False)
        
        # XXX: Add t to the sample set
        ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
        mskew = np.append(mskew, ts, axis=1)

        # Fit Regression model
        model = Ridge()
        model.fit(mskew, tYY)

        # Validate the model
        validation(mskew, tYY, model)

        if dd != './gfigs':
            with open('./mskew_feature_models/%s_ts_%s_%s_%s.pkl' % (model_name, TSTEPS, t, otype), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./mskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' % (model_name, TSTEPS, t, otype), 'wb') as f:
                pickle.dump(model, f)

def point_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./point_feature_models'):
        os.makedirs('./point_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    count = 0

    for i, s in enumerate(mms):
        for j, t in enumerate(tts):
            if count % 50 == 0:
                print('Done: ', count)
            count += 1

            # Get the data for the given moneyness and term structure
            train_vec = tX[:, :, i, j]

            # Extract features
            train_vec = extract_features(train_vec, tY[:, i, j], model_name=model_name, TSTEPS=TSTEPS, type='point', dd=dd, otype=otype)

            # XXX: Add the moneyness and term structure to the sample set
            k = np.array([s, t]*tX.shape[0]).reshape(tX.shape[0], 2)
            train_vec = np.append(train_vec, k, axis=1)

            # Fit Regression model
            reg = Ridge()
            reg.fit(train_vec, tY[:, i, j])

            # Validate the model
            validation(train_vec, tY[:, i, j], reg)

            # XXX: Save the model
            import pickle
            if dd != './gfigs':
                with open('./point_feature_models/%s_ts_%s_%s_%s_%s.pkl' %
                          (model_name, TSTEPS, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)
            else:
                with open('./point_feature_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                          (model_name, TSTEPS, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)

def run_all_models():
    for k in ['pca', 'autoencoder']:
        for j in [5, 10, 20]:
            mskew_pred(model_name=k, TSTEPS=j)
            tskew_pred(model_name=k, TSTEPS=j)
    # Do a run for the HAR method
    tskew_pred(model_name='har', TSTEPS=21)
    mskew_pred(model_name='har', TSTEPS=21)


if __name__ == "__main__":

    otype = 'call'
    dd = './figs'
    TSTEPS = 5 
    model_name = 'vae'

    data = load_data(otype, dd, TSTEPS)
    mskew_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # tskew_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # point_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # for n in ['call', 'put']:
        # for i in ['./figs', './gfigs']:
        #     for k in ['vae']:
        #         for j in [5, 10, 20]:
        #             if j == 20 and n == 'call':
        #                 continue
        #             if (j == 10 or j == 5) and n == 'call' and i == './figs':
        #                 continue
        #             print('Running for mskew: ', n, i, k, j)
        #             p = multiprocessing.Process(target=mskew_pred, args=(n, k, j, i))
        #             p.start()
        #             p.join()

        # for i in ['./figs', './gfigs']:
        #     for k in ['vae']:
        #         for j in [20]:
        #             print('Running for point: ', n, i, k, j)
        #             p = multiprocessing.Process(target=point_pred, args=(n, k, j, i))
        #             p.start()
        #             p.join()
        #             # point_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
        
        # for i in ['./figs', './gfigs']:
        #     for k in ['vae']:
        #         for j in [5, 10, 20]:
        #             if j == 20:
        #                 continue
        #             if (j == 10 or j == 5) and i == './figs' and n == 'call':
        #                 continue
        #             print('Running for tskew: ', n, i, k, j)
        #             p = multiprocessing.Process(target=tskew_pred, args=(n, k, j, i))
        #             p.start()
        #             p.join()
        #             # tskew_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
  
        # For HAR 
        # for i in ['./figs', './gfigs']:
        #     print('Running for: ', n, i, 'har')
        #     tskew_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
        #     mskew_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
        #     point_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
