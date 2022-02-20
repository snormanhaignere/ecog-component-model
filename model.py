"""Main modules are train_simple and train_complex.

Both modules fit the component model to a data matrix.

train_complex has additional functionality but as a consequence
is harder to understand / read.

train_simple implements the just the bones of main model from the paper.

Models are implemented in TensorFlow (version 1.1.0).
"""

import numpy as np
import os
import tensorflow as tf
import pickle
import re
import sys

def num2str(x):
    """Convert number to a string"""
    return re.sub(' ', '-', re.sub('[\[\],]', '', str(x)))

def mkdir(s):
    """Make a directory if it doesn't exist and return that directory"""
    if not(os.path.exists(s)):
        os.makedirs(s)
    return s

# Helper functions used to initialize weights
def weight_variable(shape, sig=0.01, seed=0):
    initial = tf.truncated_normal(shape, stddev=sig, seed=seed)
    return tf.Variable(initial)
def weight_variable_ones(shape):
    initial = tf.ones(shape, tf.float32)
    return tf.Variable(initial)
def weight_variable_zeros(shape):
    initial = tf.zeros(shape, tf.float32)
    return tf.Variable(initial)

def conv1d(k, Y):
    """Wraps tf.nn.conv1d
    
    Args:
        k: [batch x time x feature] input matrix
        Y: [time x feature x output] kernel matrix
    
    Returns: [batch x time x output]
    """

    return tf.nn.conv1d(k, Y, stride=1, padding='SAME')

def unimodal_kernel(kernel_size, K):
    """Implements constraints forcing smoothing kernel to be unimodal
    
    Args:
        kernel_size: number of samples in the kernel
        K: number of kernels (usually equal to number of components)
        
    Returns [sample x K] tensor
    """
    
    # Input timepoints
    h_t = tf.constant(2 * (np.arange(0, kernel_size) / (
        (kernel_size) - 1) - 0.5), dtype=tf.float32, shape=[kernel_size, 1])

    # The logistic function that we're going to use
    # to determine the cross-over point between
    # positive and negative derivative values
    h_mu = tf.minimum(tf.maximum(weight_variable_zeros([1, K]), -0.99), 0.99)
    h_k = tf.maximum(tf.abs(weight_variable_ones([1, K])), 0.001)
    h_l = 0.5 - 1 / (1 + tf.exp(-h_k * (h_t - h_mu)))

    # Calculate derivatives
    # x, y are the two streams / columns in Figure S7E
    h_x1 = tf.abs(weight_variable_ones([kernel_size, K]))
    h_x2 = tf.abs(weight_variable_ones([kernel_size, K]))
    h_y1 = tf.multiply(h_x1, tf.maximum(h_l, 0))
    h_y2 = tf.multiply(h_x2, tf.maximum(-h_l, 0))
    h_z1 = h_y1 / tf.reduce_sum(h_y1, axis=0)
    h_z2 = h_y2 / tf.reduce_sum(h_y2, axis=0)

    # Integrate to arrive at the final kernel
    h_final = tf.reverse(tf.cumsum(h_z1 - h_z2), axis=[0])
    
    return h_final

def train_simple(D, K, activation_penalty, 
                 activation_scale=0.001, n_iter=10000, n_iter_per_eval=20, 
                 seed=0, kernel_size=None, step_size=[0.01, 0.0032, 0.001, 0.0003, 0.0001]):
    
    """Fits a K-component model to data matrix D.
    
    Relative to the train_complex module, this code is simpler, 
    and just implements the main model described in the paper.
    
    train_complex is stricly more expressive, but the code 
    is therefore harder to follow.
    
    Args:
        D: Data matrix [sound/stimulus x time/sample x electrode/channel] 
          Note this code does not support batching, so the data
          and model must fit in memory.
        K: number of components
        activation_penalty: Penalty placed on the L1 norm of the activations.
        activation_scale: Scale of the randomly initialized activations
        kernel_size: Size of the smoothing kernel in samples.
          If None, defaults to the number of timepoints/samples (i.e., D.shape[1])
        n_iter: Total number of gradient steps to take
        step_size: List of step sizes to use.
          e.g., if [0.1, 0.01] first 50% of steps will use 0.1, last 50% will use 0.01.
          Controls the magnitude of the gradient steps used by ADAM optimizer.
        n_iter_per_eval: Number of steps to take in between loss evaluations.
          If set to 1, the loss is evaluated after every step. 
        seed: Random seed used to initialize the model
          Model should return identical results for a given seed
          To assess stability, run many times with different seeds.        
    """
    
    tf.reset_default_graph()
    
    n_stim = D.shape[0]
    n_tps = D.shape[1]
    n_elec = D.shape[2]
    
    # Size of the kernel defaults
    # to the number of timepoints
    if kernel_size is None:
        kernel_size = D.shape[1]

    # This is the exact amount of padding
    # needed to ensure causal convolution.
    pad_size = np.int32(np.floor(kernel_size / 2))

    D = np.concatenate((D, np.zeros((n_stim, pad_size, n_elec))), 1)
    d = tf.placeholder(dtype=tf.float32, shape=[
                       n_stim, n_tps + pad_size, n_elec])
        
    # Activations
    # Non-negative latent variable is constrained to be positive
    # Zeroing out of initial timepoints is needed to implement causal convolution
    A = []
    for i in range(K):
        A.append(weight_variable([n_stim, n_tps + pad_size, 1],
                           activation_scale, seed=seed))
    Amask_numpy = np.concatenate(
        (np.zeros((n_stim, pad_size, 1)), np.ones((n_stim, n_tps, 1))), axis=1)
    Amask = tf.constant(Amask_numpy, tf.float32)
    Apos = []
    for i in range(K):
        Apos.append(tf.multiply(tf.abs(A[i]), Amask))

    # Smoothing kernel
    Huni = unimodal_kernel(kernel_size, K)
    Hpos = []
    for i in range(K):
        x = tf.reshape(Huni[:, i], [kernel_size, 1, 1])
        Hpos.append(tf.abs(x) / tf.reduce_max(tf.abs(x)))

    # Electrode weights
    W = weight_variable([K, n_elec], 1, seed=seed)
    Wpos = tf.abs(W) / (K * tf.reduce_mean(tf.abs(W)))

    # Compute component responses (R) from activations and kernels
    # and then the data prediction (Y) from responses and weights
    R = []
    for i in range(K):
        R.append(conv1d(Apos[i], Hpos[i]))
    Y = tf.reshape(tf.matmul(tf.reshape(tf.concat(R[:], 2), [n_stim * (n_tps + pad_size), K]),
                             Wpos), d.shape)

    # Model loss
    loss = tf.reduce_mean(tf.square(Y - d)) + activation_penalty * tf.reduce_mean(tf.concat(Apos, 2))

    # We're now set to perform the optimization.
    with tf.Session() as sess:

        # Create a learning rate schedule
        if len(step_size) > 1:
            global_step = tf.Variable(np.int32(0), trainable=False)
            boundaries = []
            for i in range(len(step_size) - 1):
                boundaries.append(
                    np.int32(np.round((i + 1) * n_iter / len(step_size))))
            values = step_size
            learning_rate = tf.train.piecewise_constant(
                global_step, boundaries, step_size)
            train_step = tf.train.AdamOptimizer(
                learning_rate).minimize(loss)
            increment_global_step_op = tf.assign(
                global_step, global_step + 1)
        elif len(step_size) == 1: # or just set the learning rate to be constant
            train_step = tf.train.AdamOptimizer(
                step_size[0]).minimize(loss)
        else:
            raise NameError('step_size must be a list of length 1 or 2')

        sess.run(tf.global_variables_initializer())

        train_dict = {d: D}
        n_iter_per_eval = np.int32(n_iter_per_eval)
        n_evals = np.int32(np.floor((n_iter - 1) / n_iter_per_eval) + 1)
        train_loss = np.zeros((n_evals + 1, 1))
        train_loss[0] = loss.eval(feed_dict=train_dict)
        
        # Run optimization
        best_loss = 1e100
        best_loss_index = 0
        not_improved = 0
        for i in range(n_iter):

            # Check if we're going to evaluate
            # the loss this iteration
            eval_iter = np.int32(np.floor((i + 1) / n_iter_per_eval))
            if np.mod(i, np.round(n_iter / 50)) == 0:
                print(i, train_loss[eval_iter])

            # Take one gradient step
            train_step.run(feed_dict=train_dict)

            # Evaluate the loss
            if np.mod(i + 1, n_iter_per_eval) == 0:
                train_loss[eval_iter] = loss.eval(train_dict)
                
            if len(step_size) > 1:
                sess.run(increment_global_step_op)

        # Return key parameters
        A_value = tf.concat(Apos, 2).eval()
        H_value = tf.reshape(tf.concat(Hpos, 1), [kernel_size, K]).eval()
        R_value = tf.concat(R, 2).eval()
        W_value = Wpos.eval()
        Y_value = Y.eval()
        Z = {'R': R_value, 'W': W_value, 'A': A_value, 'H': H_value, 'train_loss': train_loss}

    return Z

def train_complex(D, K, activation_penalty, 
          activation_norm=1, activation_scale=0.001,
          n_iter=10000, n_iter_per_eval=20, seed=0, kernel_size=None, 
          step_size=[0.01, 0.0032, 0.001, 0.0003, 0.0001], 
          train_val_test=None, early_stopping_steps=0, 
          Hunimodal=True, Hdirac=False, nonlin='abs', 
          kernel_deriv_penalty=0, kernel_deriv_norm=2, 
          activation_deriv_penalty=0, activation_deriv_norm=2, 
          shared_kernel=False, Aval=None, Hval=None, Wval=None, log_dir=os.getcwd()):
    
    """Fits a K-component model to data matrix D.
    
    Relative to train_simple, has a bunch of extra bells and whistles.
    
    Args:
        D: Data matrix [sound/stimulus x time/sample x electrode/channel] 
          Note this code does not support batching, so the data
          and model must fit in memory.
        K: number of components
        activation_penalty: Penalty placed on the L1 (default)
          or L2 norm of the component activations. 
          This penalty is affected by the scale of the data, and thus 
          must be varied and selected by the user.
        activation_norm: Whether to use L1 or L2 norm
          Defaults to L1, thus encouraging sparsity.
        activation_scale: Scale of the randomly initialized activations
        kernel_size: Size of the smoothing kernel in samples.
          If None, defaults to the number of timepoints/samples (i.e., D.shape[1])
        n_iter: Total number of gradient steps to take
        step_size: List of step sizes to use.
          e.g., if [0.1, 0.01] first 50% of steps will use 0.1, last 50% will use 0.01.
          Controls the magnitude of the gradient steps used by ADAM optimizer.
        n_iter_per_eval: Number of steps to take in between loss evaluations.
          If set to 1, the loss is evaluated after every step. Has no effect
          on training unless early stopping is used (not the default, 
          see early_stopping_steps below).
        seed: Random seed used to initialize the model
          Model should return identical results for a given seed
          To assess stability, run many times with different seeds.         
        train_val_test: Makes it possible to only train on a subset of the data.
          If None, model is trained on all datapoints.
          Otherwise, train_val_test is a matrix with the same dimensions as the data.
          This matrix should contain one of three values: 0=train, 1=validation, 2=test
          The model will return the loss / error for the validation and test datapoints.
          There is no real difference between validation and test, unless using early 
          stopping in which case validation is used if there are validation datapoints.
        early_stopping_steps: If early_stopping_steps>0, model optimization is halted 
          if the loss has not improved over a series of N evaluations, where N=early_stopping_steps
          If there are validation datapoints, these datapoints are used to decide whether to stop.
        nonlin: The nonlinear function used to impose non-negativity on real-valued latents
          Default: absolutue value ('abs')
          Other options: 'relu', 'softplus', 'square', 'continuous-huber'
        Hunimodal: If True (the default), forces smoothing kernel to be unimodal.
        Hdirac: If True, sets smoothing kernel to dirac delta function, thus eliminating its influence
        kernel_deriv_penalty: Optional penalty to be placed on L1 or L2 norm of kernel derivative
        kernel_deriv_norm: Whether to penalize L1 or L2 norm of kernel derivative
        activation_deriv_penalty: Optional penalty to be placed on L1 or L2 norm of activations
        activation_deriv_norm: Whether to penalize the L1 or L2 norm of the activation derivative
        shared_kernel: If True, forces all components to share the same smoothing kernel
        Aval, Hval, Wval: makes it possible to specify/fix the value of the 
          activations (Val), electrode weights (Wval), or smoothing kernel (Hval)
        log_dir: Directory to save checkpoints to (only relevant if using early stopping)        
    """
    
    tf.reset_default_graph()
    
    n_stim = D.shape[0]
    n_tps = D.shape[1]
    n_elec = D.shape[2]
    
    # Size of the kernel defaults
    # to the number of timepoints
    if kernel_size is None:
        kernel_size = D.shape[1]

    # This is the exact amount of padding
    # needed to ensure causal convolution.
    pad_size = np.int32(np.floor(kernel_size / 2))

    D = np.concatenate((D, np.zeros((n_stim, pad_size, n_elec))), 1)
    d = tf.placeholder(dtype=tf.float32, shape=[
                       n_stim, n_tps + pad_size, n_elec])

    # Figure out if there is separate
    # training, validation, and testing data,
    # and create a placeholder for this information.
    if not(train_val_test is None):

        if np.sum(train_val_test==1)>0:
            validate = True
        else:
            validate = False

        if np.sum(train_val_test==2)>0:
            test = True
        else:
            test = False

        if (np.ndim(train_val_test) == 2) and (train_val_test.shape[0] == n_stim) and (train_val_test.shape[1] == n_elec):
            train_val_test = np.repeat(np.expand_dims(train_val_test, axis=1),
                             n_tps + pad_size, axis=1)
        else:

            assert(train_val_test.shape == (n_stim, n_tps, n_elec))
            train_val_test = np.concatenate(
                (train_val_test, np.zeros((n_stim, pad_size, n_elec))), 1)
        tfmask = tf.placeholder(dtype=tf.float32, shape=[
                                n_stim, n_tps + pad_size, n_elec])

    else:
        validate = False
        test = False
        train_val_test = np.zeros([n_stim, n_tps + pad_size, n_elec], dtype=np.float32)
        tfmask = tf.placeholder(dtype=tf.float32, shape=[n_stim, n_tps + pad_size, n_elec])

    # Function to convert real-valued latent variables
    # to positive variables.
    if nonlin == 'abs':
        nonlin_fn = tf.abs
    elif nonlin == 'relu':
        nonlin_fn = tf.nn.relu
    elif nonlin == 'softplus':
        nonlin_fn = tf.nn.softplus
    elif nonlin == 'square':
        nonlin_fn = tf.square
    elif nonlin == 'continuous-huber':
        nonlin_fn = lambda x: tf.sqrt(1 + x**2) - 1
    else:
        raise NameError('No matching nonlinearity')

    # Initialize the component activations
    if (Aval is None):

        # Initialize to real numbers
        A = []
        for i in range(K):
            A.append(weight_variable([n_stim, n_tps + pad_size, 1],
                               activation_scale, seed=seed))

        # Zero the initial pad_size values to force
        # causal convolution. Do this by creating a mask
        # with zeros at the front and ones at then end.
        Amask_numpy = np.concatenate(
            (np.zeros((n_stim, pad_size, 1)), np.ones((n_stim, n_tps, 1))), axis=1)
        Amask = tf.constant(Amask_numpy, tf.float32)

        # Convert reals to non-negative values and multiply by mask
        Apos = []
        for i in range(K):
            Apos.append(tf.multiply(nonlin_fn(A[i]), Amask))

    else:  # Or fix them to a prespecified value (Aval)

        Apos = []
        for i in range(K):
            Apos.append(tf.constant(Aval[:, :, i], dtype=tf.float32, shape=[
                        n_stim, n_tps + pad_size, 1]))

    # Initialize the temporal smoothing kernels
    if (Hval is None):

        H = []
        
        # Can optionally force the components
        # to all use the same kernel
        if shared_kernel:
            num_kernels = 1
        else:
            num_kernels = K
            
        for i in range(num_kernels):
            # Can optionally force kernel to be dirac delta function,
            # which has effectively nullifies the kernel.
            if Hdirac: 
                dirac = np.concatenate(
                    (np.zeros((kernel_size - 1, 1, 1)), np.ones((1, 1, 1))), axis=0)
                H.append(tf.constant(dirac, dtype=tf.float32,
                                     shape=[kernel_size, 1, 1]))
            else:
                if Hunimodal:
                    # We initialize all kernels at once
                    # and so we only do this during the first loop.
                    if i == 0:
                        Huni = unimodal_kernel(kernel_size, K)
                    H.append(tf.reshape(
                        Huni[:, i], [kernel_size, 1, 1]))
                else:
                    H.append(weight_variable(
                            [kernel_size, 1, 1], 1, seed=seed))

        # Force kernel to be positive (only relevant if not unimodal)
        # and normalize by max value
        Hpos = []
        for i in range(K):
            if shared_kernel:
                xi = 0
            else:
                xi = i
            Hpos.append(nonlin_fn(H[xi]) / tf.reduce_max(nonlin_fn(H[xi])))
            
    else: # Or fix them to a prespecified value (Hval)

        Hpos = []
        for i in range(K):
            Hpos.append(tf.constant(
                Hval[:, i], dtype=tf.float32, shape=[kernel_size, 1, 1]))

    R = []
    for i in range(K):
        R.append(conv1d(Apos[i], Hpos[i]))

    if (Wval is None):

        W = weight_variable([K, n_elec], 1, seed=seed)
        Wpos = nonlin_fn(W) / (K * tf.reduce_mean(nonlin_fn(W)))

    else:

        Wpos = tf.constant(Wval, dtype=tf.float32, shape=[K, n_elec])

    # Prediction: R * W
    # Need to do some reshaping shenanigans to implement this
    Y = tf.reshape(tf.matmul(tf.reshape(tf.concat(R[:], 2), [n_stim * (n_tps + pad_size), K]),
                             Wpos), d.shape)

    # Loss: reconstruction error
    if train_val_test is None:
        prediction_error = tf.reduce_mean(tf.square(Y - d))
    else:
        prediction_error = tf.reduce_sum(tf.square(Y - d) * tfmask) / tf.reduce_sum(tfmask)

    # Loss: L1 or L2 penalty on activations
    if activation_norm == 1:
        activation_error = activation_penalty * tf.reduce_mean(tf.concat(Apos, 2))
    elif activation_norm == 2:
        activation_error = activation_penalty * tf.sqrt(tf.reduce_mean(tf.square(tf.concat(Apos, 2))))
    else:
        raise NameError('Norm must be L1 or L2')

    # Loss: kernel smoothness
    Hcat = tf.concat(Hpos, 2)
    if kernel_deriv_norm == 1:
        unpenalized_error = tf.reduce_mean(
            tf.abs(Hcat[0:kernel_size - 1, :, :] - Hcat[1:kernel_size, :, :]))
    elif kernel_deriv_norm == 2:
        unpenalized_error = tf.reduce_mean(
            tf.square(Hcat[0:kernel_size - 1, :, :] - Hcat[1:kernel_size, :, :]))
    else:
        raise NameError('No valid smoothness penalty function')
    basis_smoothness_error = kernel_deriv_penalty * unpenalized_error

    # Loss: activation smoothness
    activation_diff = (tf.concat(Apos, 2)[:, pad_size:n_tps + pad_size - 1, :]
                       - tf.concat(Apos, 2)[:, pad_size + 1:n_tps + pad_size, :])   
    if activation_deriv_norm == 1:
        activation_smoothness_error = activation_deriv_penalty * \
            tf.reduce_mean(
                tf.reduce_mean(tf.abs(activation_diff), axis=(0, 1)))
    elif activation_deriv_norm == 2:
        activation_smoothness_error = activation_deriv_penalty * \
            tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(activation_diff), axis=(0, 1))))
    else:
        raise NameError('Activation smoothness norm must be L1 or L2')

    # Sum all of the losses together
    loss = prediction_error + activation_error + \
        basis_smoothness_error + activation_smoothness_error

    # We're now set to perform the optimization.
    with tf.Session() as sess:

        # Create a learning rate schedule
        if len(step_size) > 1:
            global_step = tf.Variable(np.int32(0), trainable=False)
            boundaries = []
            for i in range(len(step_size) - 1):
                boundaries.append(
                    np.int32(np.round((i + 1) * n_iter / len(step_size))))
            values = step_size
            learning_rate = tf.train.piecewise_constant(
                global_step, boundaries, step_size)
            train_step = tf.train.AdamOptimizer(
                learning_rate).minimize(loss)
            increment_global_step_op = tf.assign(
                global_step, global_step + 1)
        elif len(step_size) == 1: # or just set the learning rate to be constant
            train_step = tf.train.AdamOptimizer(
                step_size[0]).minimize(loss)
        else:
            raise NameError('step_size must be a list of length 1 or 2')

        sess.run(tf.global_variables_initializer())

        # Training dictionary and loss
        if train_val_test is None:
            train_dict = {d: D}
        else:
            train_dict = {d: D, tfmask: np.float32(train_val_test==0)}
            print('Training timeponts:', np.sum(np.float32(train_val_test==0)))
        n_iter_per_eval = np.int32(n_iter_per_eval)
        n_evals = np.int32(np.floor((n_iter - 1) / n_iter_per_eval) + 1)
        train_loss = np.zeros((n_evals + 1, 1))
        train_loss[0] = loss.eval(feed_dict=train_dict)
        train_prederr = np.zeros((n_evals + 1, 1))
        train_prederr[0] = prediction_error.eval(feed_dict=train_dict)

        # Validation dictionary and loss
        if validate:
            val_dict = {d: D, tfmask: np.float32(train_val_test==1)}
            print('Validation timeponts:', np.sum(np.float32(train_val_test==1)))
            val_loss = np.zeros((n_evals + 1, 1))
            val_loss[0] = loss.eval(feed_dict=val_dict)
            val_prederr = np.zeros((n_evals + 1, 1))
            val_prederr[0] = prediction_error.eval(feed_dict=val_dict)

        # Test dictionary and loss
        if test:
            test_dict = {d: D, tfmask: np.float32(train_val_test==2)}
            print('Test timeponts:', np.sum(np.float32(train_val_test==2)))
            test_loss = np.zeros((n_evals + 1, 1))
            test_loss[0] = loss.eval(feed_dict=test_dict)
            test_prederr = np.zeros((n_evals + 1, 1))
            test_prederr[0] = prediction_error.eval(feed_dict=test_dict)                

        if early_stopping_steps > 0:
            saver = tf.train.Saver(max_to_keep=1)
            checkpt_filename = log_dir + '/seed' + str(seed) + '.ckpt'

        # Run optimization
        best_loss = 1e100
        best_loss_index = 0
        not_improved = 0
        for i in range(n_iter):

            # Check if we're going to evaluate
            # the loss this iteration
            eval_iter = np.int32(np.floor((i + 1) / n_iter_per_eval))
            if np.mod(i, np.round(n_iter / 50)) == 0:
                print(i, train_loss[eval_iter])

            # Take one gradient step
            train_step.run(feed_dict=train_dict)

            # Evaluate the loss
            if np.mod(i + 1, n_iter_per_eval) == 0:

                train_loss[eval_iter] = loss.eval(train_dict)
                train_prederr[eval_iter] = prediction_error.eval(train_dict)
                if validate:
                    val_loss[eval_iter] = loss.eval(val_dict)
                    val_prederr[eval_iter] = prediction_error.eval(val_dict)
                if test:
                    test_loss[eval_iter] = loss.eval(test_dict)
                    test_prederr[eval_iter] = prediction_error.eval(test_dict)

                # Early stopping / saving
                if early_stopping_steps > 0:

                    # Prefer to use validation loss for early stopping
                    if validate:
                        stop_loss = val_prederr[eval_iter]
                    else:
                        stop_loss = train_loss[eval_iter]

                    # Check if loss has improved, if so save
                    if stop_loss < best_loss:
                        best_loss = stop_loss
                        best_loss_index = eval_iter
                        saver.save(sess, checkpt_filename)
                        not_improved = 0
                    else:
                        not_improved += 1

                    # Check if we should stop
                    if not_improved == early_stopping_steps:
                        train_loss = train_loss[:(eval_iter+1)]
                        train_prederr = train_prederr[:(eval_iter+1)]
                        if validate:
                            val_loss = val_loss[:(eval_iter+1)]
                            val_prederr = val_prederr[:(eval_iter+1)]
                        if test:
                            test_loss = test_loss[:(eval_iter+1)]
                            test_prederr = test_prederr[:(eval_iter+1)]
                        break

            if len(step_size) > 1:
                sess.run(increment_global_step_op)

        # Load the best loss
        if early_stopping_steps > 0:
            saver.restore(sess, checkpt_filename)

        # Return key parameters
        A_value = tf.concat(Apos, 2).eval()
        H_value = tf.reshape(tf.concat(Hpos, 1), [kernel_size, K]).eval()
        R_value = tf.concat(R, 2).eval()
        W_value = Wpos.eval()
        Y_value = Y.eval()
        prediction_error_value = prediction_error.eval(feed_dict=train_dict)
        activation_error_value = activation_error.eval()
        basis_smoothness_error_value = basis_smoothness_error.eval()
        activation_smoothness_error = activation_smoothness_error.eval()
        activation_smoothness_weights = sw_pos.eval()
        Z = {'R': R_value, 'W': W_value, 'A': A_value, 'H': H_value,
            'train_loss': train_loss, 'train_prederr': train_prederr, 
            'prediction_error': prediction_error_value,
            'activation_error': activation_error_value,
            'basis_smoothness_error': basis_smoothness_error_value, 'pad_size': pad_size,
            'activation_smoothness_weights': activation_smoothness_weights,
            'activation_smoothness_error': activation_smoothness_error}

        if validate:
            Z['val_loss'] = val_loss
            Z['val_prederr'] = val_prederr

        if test:
            Z['test_loss'] = test_loss
            Z['test_prederr'] = test_prederr

        if early_stopping_steps > 0:
            Z['best_loss'] = best_loss
            Z['best_loss_index'] = best_loss_index

    return Z



