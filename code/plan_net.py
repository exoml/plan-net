# Code for plan-net
# Adam D. Cobb, Feb 2019

# Keras
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.metrics import binary_accuracy
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input, Lambda, Wrapper, merge, concatenate
from keras.engine import InputSpec
from keras.layers.core import Dense, Dropout, Activation, Layer, Lambda, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adadelta, adam
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
import tensorflow as tf

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn import metrics, neighbors
from sklearn.preprocessing import MinMaxScaler


# From: https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb
class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


# This is the BNN class which learns cholesky. 
class BNNModel_het_chol:
    """
    Builds basic BNN model around training data
    """
    
    def __init__(self, X: np.array, Y: np.array, architecture: list, dropout = 0.1, T = 10,
                 tau = 1.0, lengthscale = 1., base_lr = 5e-2, gamma = 0.0001*0.25, ens_num = 0, train_flag = True):
        """
        :X: training data X -> so far only implemented for 1D data, needs to be of shape (n,1) or (1,n)
        :Y: training data y, needs to be passed as array of shape (n,1);
        :param architecture: list of perceptrons per layer, as long as network deep
        :param dropout: probability of perceptron being dropped out
        :param T: number of samples from posterior of weights during test time
        :param tau: precision of prior
        :param lengthscale: lengthscale
        :param base_lr: initial learning rate for SGD optimizer
        :param gamma: parameter for decay of initial learning rate according to default SGD learning schedule
        """
        if np.shape(X)[0] == len(Y):
            assert np.shape(X)[1] >= 1
        else:
            assert np.shape(X)[1] == len(Y)
            X = np.transpose(X)
            
        self.X = X
#         assert np.shape(Y)[1] == 1
        self.Y = Y
        D = self.Y.shape[-1]
        
        self.ens_num = ens_num
        self.dropout = dropout
        self.T = T
        self.tau = tau
        self.lengthscale = lengthscale 
        # Eq. 3.17 Gal thesis:
        self.weight_decay = ((1-self.dropout)*self.lengthscale**2)/(self.X.shape[0]*self.tau) # Don't need to dived by two as we are using squared error
        self.architecture = architecture
        self.train_flag = train_flag

        if K.backend() == 'tensorflow':
            K.clear_session()
        N = self.X.shape[0]
        wd = self.lengthscale**2. / N
        dd = 2. / N
        inp = Input(shape=(np.shape(self.X)[1],))
        x = inp
        x = ConcreteDropout(Dense(architecture[0], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        for jj in range(1,(len(architecture))):
            x = ConcreteDropout(Dense(architecture[jj], activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        mean = ConcreteDropout(Dense(self.Y.shape[-1]), weight_regularizer=wd, dropout_regularizer=dd)(x)
        log_var = ConcreteDropout(Dense(int(D * (D+1)/2)), weight_regularizer=wd, dropout_regularizer=dd)(x)
        out = concatenate([mean, log_var])
        self.model = Model(inp, out)
    
        
        def heteroscedastic_loss(true, pred):
            mean = pred[:, :D]
            L = pred[:, D:]
            N = tf.shape(true)[0]
            # Slow:
            k = 1
            inc = 0
            Z = []
            diag = []
            for d in range(D):
            #         for j in range(k):
#                 L[:,k-1] = K.exp(L[:,k-1]) # constrain diagonal to be positive
                if k == 1:
                    Z.append(tf.concat([tf.exp(tf.reshape(L[:,inc:inc+k],[N,k])),tf.zeros((N,D-k))],1))
                else:
                    Z.append(tf.concat([tf.reshape(L[:,inc:inc+k-1],[N,k-1]),tf.exp(tf.reshape(L[:,inc+k-1],[N,1])),tf.zeros((N,D-k))],1))
                diag.append(K.exp(L[:,inc+k-1]))
                inc += k
                k+=1
            diag = tf.concat(tf.expand_dims(diag,-1),-1)
            lower = tf.reshape(tf.concat(Z,-1),[N,D,D])
            S_inv = tf.matmul(lower,tf.transpose(lower,perm=[0,2,1]))
            x = tf.expand_dims((true - mean),-1)
            quad = tf.matmul(tf.matmul(tf.transpose(x,perm=[0,2,1]),S_inv),x)
            log_det = - 2 * K.sum(K.log(diag),0)
            # - 0.5 * [log det + quadratic term] = log likelihood 
            # remove minus sign as we want to minimise NLL
            return K.mean(tf.squeeze(quad,-1) + log_det, 0)



        self.model.compile(optimizer='adam', loss=heteroscedastic_loss)
#         assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
#         assert len(model.losses) == 5  # a loss for each Concrete Dropout layer
#         hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
#         loss = hist.history['loss'][-1]
#         return model, -0.5 * loss  # return ELBO up to const.

    
    def train(self, epochs = 100, batch_size = 128, validation_data = ()):
        """
        Trains model
        :param epochs: defines how many times each training point is revisited during training time
        :param batch_size: defines how big batch size used is
        """
        # Might want to save model check points?!
        weights_file_std = './ens_folder_models/ensemble_'+str(self.ens_num)+'_check_point_weights_het_loss.h5'
        model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                   save_weights_only=True, mode='auto',verbose=0)

        Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
        if self.train_flag:
            self.historyBNN = self.model.fit(self.X, self.Y, epochs=epochs,
                                         batch_size=batch_size, verbose=2,
                                         validation_data = validation_data, callbacks=[Early_Stop,model_checkpoint])
        self.model.load_weights(weights_file_std)
        #        tl,vl = historyBNN.history['loss'], historyBNN.history['val_loss'] 
        
    def predict(self, X_test):
        D = self.Y.shape[-1]
        Yt_hat = np.array([self.model.predict(X_test, batch_size=500, verbose=0) for _ in range(self.T)])
#         Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        mean = Yt_hat[:, :, :D]  # K x N x D
        logvar = Yt_hat[:, :, D:]
        MC_pred = np.mean(mean, 0)
        return MC_pred, mean, logvar
    
    def evaluate(self, x_test, y_test):
#         rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5
        _, mean, logvar = self.predict(x_test)
        # We compute the test log-likelihood
        LL = np.zeros((x_test.shape[0],mean.shape[0]))
        for t in range(mean.shape[0]):
            Z = []
            diag = []
            inc = 0
            k=1
            N = x_test.shape[0]
            D = y_test.shape[1]
            for d in range(D):
            #         for j in range(k):
                logvar[t,:,k-1] = np.exp(logvar[t,:,k-1]) # constrain diagonal to be positive
                Z.append(np.hstack([np.reshape(logvar[t,:,inc:inc+k],[N,k]),np.zeros((N,D-k))]))
                diag.append(logvar[t,:,k-1])
                inc += k
                k+=1
            diag = np.hstack(np.expand_dims(diag,-1))
            lower = np.reshape(np.hstack(Z),[N,D,D])


            S_inv = np.matmul(lower,np.transpose(lower,axes=[0,2,1]))
            x = np.expand_dims(((np.squeeze(mean[t]) - y_test)**2),-1)
            quad = np.matmul(np.matmul(np.transpose(x,axes=[0,2,1]),S_inv),x)
            log_det = np.sum(- np.log(diag**2),1)
            # - 0.5 * [log det + quadratic term] = log likelihood 
            # remove minus sign as we want to minimise NLL
            LL[:,t] = np.squeeze(quad) + log_det

        test_ll = np.sum(np.sum(LL,-1),-1)
        rmse = np.mean((np.mean(mean, 0) - y_test)**2.)**0.5
        return test_ll/N, rmse


def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

# HELA PLOTTING FUNCTION taken from https://github.com/exoclime/HELA




def posterior_matrix(estimations, y, names, ranges, colors, soft_colors=None):
    
    cmaps = [LinearSegmentedColormap.from_list("MyReds", [(1, 1, 1), c], N=256)
             for c in colors]
    
    ranges = np.array(ranges)
    
    if soft_colors is None:
        soft_colors = colors
    
    num_dims = estimations.shape[1]
    
    fig, axes = plt.subplots(nrows=num_dims, ncols=num_dims,
                             figsize=(2 * num_dims, 2 * num_dims))
    fig.subplots_adjust(left=0.07, right=1-0.05,
                        bottom=0.07, top=1-0.05,
                        hspace=0.05, wspace=0.05)
    
    for ax, dims in zip(axes.flat, product(range(num_dims), range(num_dims))):
        dims = list(dims[::-1])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_visible(False)
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_visible(True)
            if names is not None:
                ax.set_ylabel(names[dims[1]], fontsize=18)
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_visible(True)
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_visible(True)
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_visible(True)
            if names is not None:
                ax.set_xlabel(names[dims[0]], fontsize=18)
        if ax.is_first_col() and ax.is_first_row():
            ax.yaxis.set_visible(False)
            ax.set_ylabel("")
        if ax.is_last_col() and ax.is_last_row():
            ax.yaxis.set_visible(False)
        
        if dims[0] < dims[1]:
            # locations, kd_probs, *_ = _kernel_density_joint(estimations[:, dims], ranges[dims])
            # ax.contour(locations[0], locations[1],
            #            kd_probs,
            #            colors=colors[dims[0]],
            #            linewidths=0.5
            #            # 'copper', # 'hot', 'magma' ('copper' with white background)
            #           )
            histogram, grid_x, grid_y = _histogram(estimations[:, dims], ranges[dims])
            ax.pcolormesh(grid_x, grid_y, histogram, cmap=cmaps[dims[0]])
            
            expected = np.median(estimations[:, dims], axis=0)
            ax.plot([expected[0], expected[0]], [ranges[dims[1]][0], ranges[dims[1]][1]], '-', linewidth=1, color='#222222')
            ax.plot([ranges[dims[0]][0], ranges[dims[0]][1]], [expected[1], expected[1]], '-', linewidth=1, color='#222222')
            ax.plot(expected[0], expected[1], '.', color='#222222')
            ax.axis('normal')
            if y is not None:
                real = y[dims]
                ax.plot(real[0], real[1], '*', markersize=10, color='#FF0000')
            ax.axis([ranges[dims[0]][0], ranges[dims[0]][1],
                     ranges[dims[1]][0], ranges[dims[1]][1]])
        elif dims[0] > dims[1]:
            ax.plot(estimations[:, dims[0]], estimations[:, dims[1]], '.', color=soft_colors[dims[1]])
            ax.axis([ranges[dims[0]][0], ranges[dims[0]][1],
                     ranges[dims[1]][0], ranges[dims[1]][1]])
        else:
            histogram, bins = _histogram(estimations[:, dims[:1]], ranges=ranges[dims[:1]])
            ax.bar(bins[:-1], histogram, color=soft_colors[dims[0]], width=bins[1]-bins[0])
            
            kd_probs = histogram
            expected = np.median(estimations[:, dims[0]])
            ax.plot([expected, expected], [0, 1.1 * kd_probs.max()], '-', linewidth=1, color='#222222')
            
            if y is not None:
                real = y[dims[0]]
                ax.plot([real, real], [0, kd_probs.max()], 'r-')
            ax.axis([ranges[dims[0]][0], ranges[dims[0]][1],
                     0, 1.1 * kd_probs.max()])
    
    # fig.tight_layout(pad=0)
    return fig


def _min_max_scaler(ranges, feature_range=(0, 100)):
    res = MinMaxScaler()
    res.data_max_ = ranges[:, 1]
    res.data_min_ = ranges[:, 0]
    res.data_range_ = res.data_max_ - res.data_min_
    res.scale_ = (feature_range[1] - feature_range[0]) / (ranges[:, 1] - ranges[:, 0])
    res.min_ = -res.scale_ * res.data_min_
    res.n_samples_seen_ = 1
    res.feature_range = feature_range
    return res


def _kernel_density_joint(estimations, ranges, bandwidth=1/25):
    
    ndims = len(ranges)
    
    scaler = _min_max_scaler(ranges, feature_range=(0, 100))
    
    bandwidth = bandwidth * 100
    # step = 1.0
    
    kd = neighbors.KernelDensity(bandwidth=bandwidth).fit(scaler.transform(estimations))
    locations1d = np.arange(0, 100, 1)
    locations = np.reshape(np.meshgrid(*[locations1d] * ndims), (ndims, -1)).T
    kd_probs = np.exp(kd.score_samples(locations))
    
    shape = (ndims,) + (len(locations1d),) * ndims
    locations = scaler.inverse_transform(locations)
    locations = np.reshape(locations.T, shape)
    kd_probs = np.reshape(kd_probs, shape[1:])
    return locations, kd_probs, kd


def _histogram(estimations, ranges, bins=30):
    
    if len(ranges) == 1:
        histogram, edges = np.histogram(estimations[:, 0], bins=bins, range=ranges[0])
        return histogram, edges
    
    if len(ranges) == 2:
        histogram, xedges, yedges = np.histogram2d(estimations[:, 0], estimations[:, 1], bins=bins, range=ranges)
        grid_x, grid_y = np.meshgrid(xedges, yedges)
        return histogram.T, grid_x, grid_y, 