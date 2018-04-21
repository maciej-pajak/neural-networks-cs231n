package pl.maciejpajak.casestudy;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A model of two layer fully-connected neural network.
 * Network training runs with a softmax loss function and L2 regularization.
 * The network uses a ReLU nonlinearity after the first fully connected layer
 *
 * Architecture:
 * input - fully connected layer - ReLU - fully connected layer - softmax
 *
 * The outputs of the second fully-connected layer are the scores for each class.
 *
 * Based on cs231n.
 */
public class TwoLayerNetworkCaseStudy {

    // first layer weights D x H
    private final INDArray weightsOne;
    // first layer biases 1 x H
    private final INDArray biasesOne;

    // second layer weights H x C
    private final INDArray weightsTwo;
    // second layer biases 1 x C
    private final INDArray biasesTwo;

    /**
     * Initialize the model. Weights are initialized to small random values and
     * biases are initialized to zero.
     *
     * @param inputSize - the dimension D of input data.
     * @param hiddenSize - the number of neurons H in the hidden layer.
     * @param outputSize - the number of classes.
     * @param std - random weights initialization coefficient.
     */
    public TwoLayerNetworkCaseStudy(int inputSize, int hiddenSize, int outputSize, double std) {
        weightsOne = Nd4j.randn(inputSize, hiddenSize).mul(std);
        biasesOne = Nd4j.zeros(1, hiddenSize);
        weightsTwo = Nd4j.randn(hiddenSize, outputSize).mul(std);
        biasesTwo = Nd4j.zeros(1, outputSize);
    }

    /**
     * Compute the loss and gradients for a two layer fully connected neural network.
     * @param dataSet - input data of shape N x D. Each row is a training example.
     * @param dataLables - array with training labels of shape 1 x N.
     * @param reg - regularization strength.
     *
     * @return
     */
    private Pair<Double, INDArray> loss(INDArray dataSet, INDArray dataLables, double reg) {


        return null;
    }

    /**
     * Train this neural network using stochastic gradient descent.
     *
     * @param trainingData - an array of shape N x D giving training data.
     * @param trainingLables - an array of shape N x 1 giving training labels.
     * @param validationData - an array of shape N_VAL x D giving validation data.
     * @param validationLables - an array of shape N_VAL x D giving validation labels.
     * @param learningRate - rate of optimization.
     * @param learningRateDecay - factor used to decay the learning rate after each epoch.
     * @param reg - regularization strength.
     * @param iterations - number of iterations.
     * @param batchSize - number of training examples to use per step.
     */
    public void train(INDArray trainingData, INDArray trainingLables, INDArray validationData, INDArray validationLables,
                      double learningRate, double learningRateDecay, double reg, double iterations, double batchSize) {
        throw new UnsupportedOperationException("Not yet implemented.");
    }

    /**
     * Use the trained weights of this two-layer network to predict labels for data points.
     *
     * @param dataSet - array of shape N x D containing N elements (of dimensionality D) to classify.
     * @return - array of shape N x 1 giving predicted labels for each input element.
     */
    public INDArray predict(INDArray dataSet) {
        throw new UnsupportedOperationException("Not yet implemented.");
    }

}
