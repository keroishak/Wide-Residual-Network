import tensorflow as tf
import numpy as np
import cifar100_input as data_input
import os.path

DECAY_KEY='DECAY'
Data_DIR='./cifar-100-binary'
MODEL_DIR='./model'
MODEL_File='WRNModel.ckpt'

k = 2
weightDecay = 0.0001
classes = 100
learningRate = 0.1
momentum = 0.9
batchSize = 100
trainingIterations=10000

def batchNormalization(x, isTrain, name='bn'):

    with tf.variable_scope(name):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer, trainable=False)
        sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer, trainable=False)
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer)
        update = 0.1

        updateMu = mu.assign_sub(update*(mu - batch_mean))
        updateSigma = sigma.assign_sub(update*(sigma - batch_var))

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updateMu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updateSigma)

        mean, var = tf.cond(isTrain, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

    return bn

def convLayer(x, kernelSize, outChannels, stride, pad='SAME',name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [kernelSize, kernelSize, in_shape[3], outChannels],tf.float32,
                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / kernelSize / kernelSize / outChannels)))
        if kernel not in tf.get_collection(DECAY_KEY):
            tf.add_to_collection(DECAY_KEY, kernel)
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
    return conv

def getWRNGraph(isTrain, images, labels):

    filters = [16, 16 * k, 32 * k, 64 * k]
    strides = [1, 2, 2]
    globalStep = tf.Variable(0, trainable=False, name='global_step')
    x = convLayer(images, 3, 16, 1)  # initial conv

    for i in xrange(1, 4, 1):
        with tf.variable_scope('block_%d' % i) as scope:
            x = batchNormalization(x, isTrain, name='bn_init_%d' % i)
            x = tf.nn.relu(x, name='relu')

            shortcut = convLayer(x, 1, filters[i], strides[i - 1], name='skip')

            # Residual
            x = convLayer(x, 3, filters[i], strides[i - 1], name='conv1')
            x = batchNormalization(x, isTrain, name='bn2')
            x = tf.nn.relu(x, name='relu')
            x = convLayer(x, 3, filters[i], 1, name='conv2')

            x = tf.add(x, shortcut, name='sum1')

            shortcut = x

            # Residual
            x = batchNormalization(x, isTrain, name='bn1_1')
            x = tf.nn.relu(x, name='relu_1_1')
            x = convLayer(x, 3, filters[i], 1, name='conv1_1')
            x = batchNormalization(x, isTrain, name='bn2_2')
            x = tf.nn.relu(x, name='relu_2_1')
            x = convLayer(x, 3, filters[i], 1, name='conv2_1')

            x = tf.add(x, shortcut, name='sum2')

    x = batchNormalization(x, isTrain)
    x = tf.nn.relu(x, name='relu')
    x = tf.reduce_mean(x, [1, 2])
    xShape = x.get_shape().as_list()
    x = tf.reshape(x, [-1, xShape[1]])
    w = tf.get_variable('weights', [x.get_shape()[1], classes],
                        tf.float32, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1.0 / classes)))

    b = tf.get_variable('biases', [classes], tf.float32,
                        initializer=tf.constant_initializer(0.0))
    fc = tf.nn.bias_add(tf.matmul(x, w), b)
    if fc not in tf.get_collection(DECAY_KEY):
        tf.add_to_collection(DECAY_KEY, fc)

    predictedClass=tf.argmax(fc)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc, labels=labels)
    loss = tf.reduce_mean(loss, name='cross_entropy')
    #tf.summary.scalar('cross_entropy', loss)

    l2Loss = [tf.nn.l2_loss(w) for w in tf.get_collection(DECAY_KEY)]
    l2Loss = tf.multiply(weightDecay, tf.add_n(l2Loss))

    totalLoss = loss + l2Loss

    learningDecay = tf.train.exponential_decay(learningRate, globalStep,
                                               1000 * 5, learningRate, staircase=True)

    optimizer = tf.train.MomentumOptimizer(learningDecay, momentum)
    gradsAndVars = optimizer.compute_gradients(totalLoss, tf.trainable_variables())
    applyGradOperation = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

    updateOperations = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if updateOperations:
        with tf.control_dependencies(updateOperations + [applyGradOperation]):
            trainOperations = tf.no_op()
    else:
        trainOperations = applyGradOperation


    return trainOperations, predictedClass, loss

def train():

    images = tf.placeholder(tf.float32, [batchSize, data_input.IMAGE_SIZE, data_input.IMAGE_SIZE, 3])
    labels = tf.placeholder(tf.int32, [batchSize])

    isTrain = tf.placeholder(tf.bool)
    with tf.variable_scope('train_image'):
        trainImages, trainLabels = data_input.distorted_inputs(Data_DIR, 100)

    trainOperations, predictedClass, loss = getWRNGraph(isTrain, images, labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()

        for iteration in range(trainingIterations):
            trainImagesVal, trainLabelsVal = sess.run([trainImages, trainLabels])

            _, lossValue = sess.run([trainOperations, loss],feed_dict=
                {isTrain: True, images: trainImagesVal, labels: trainLabelsVal})

            if iteration % 1000 == 0:
                print "iteration %d with loss = %f" % (iteration,lossValue)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess,MODEL_DIR+'/'+MODEL_File)

def test():

    tf.reset_default_graph()
    images = tf.placeholder(tf.float32, [batchSize, data_input.IMAGE_SIZE, data_input.IMAGE_SIZE, 3])
    labels = tf.placeholder(tf.int32, [batchSize])

    isTrain = tf.placeholder(tf.bool)
    with tf.variable_scope('test_image'):
        testImages, testLabels = data_input.inputs(True ,Data_DIR , batchSize)
        _, predictedClass, loss = getWRNGraph(isTrain, images, labels)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        saver.restore(sess,MODEL_DIR+'/'+MODEL_File)
        correctlyClassified=0
        misClassified=0
        for example in range(100):

            testImagesVal, testLabelsVal = sess.run([testImages, testLabels])
            Predclass, lossValue = sess.run([predictedClass, loss],feed_dict=
                {isTrain: False, images: testImagesVal, labels: testLabelsVal})

            for i in range(batchSize):
                if Predclass[i]==testLabelsVal[i]:
                    correctlyClassified+=1
                else:
                    misClassified+=1
        coord.request_stop()
        coord.join(threads)
        print "accuracy = %f" % (correctlyClassified/(correctlyClassified+misClassified))

def main():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    train()
    test()
    #if os.path.isfile(MODEL_DIR+'/'+MODEL_File)==False:

    #test()
if __name__ == '__main__':
    main()