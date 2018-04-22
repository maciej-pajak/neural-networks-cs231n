package pl.maciejpajak.casestudy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import pl.maciejpajak.classifier.Nd4jHelper;

import java.util.logging.Logger;

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

    private final static Logger LOG = Logger.getLogger(TwoLayerNetworkCaseStudy.class.getName());

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

        int samples = trainingData.size(0);

        for (int i = 0 ; i < iterations ; i++) {
            // evaluate class scores N x C
            INDArray hiddenLayer = Transforms.max(trainingData.mmul(weightsOne).add(biasesOne), 0); // ReLU activation
            INDArray scores = hiddenLayer.mul(weightsTwo).add(biasesTwo);

            // compute the class probabilities
            INDArray expScores = Transforms.exp(scores);
            INDArray probs = expScores.divColumnVector(expScores.sum(1));

            // compute hte loss: average cross-entropy loss and regularization
            INDArray correctLogProbs = Transforms.log(probs).mul(-1);

            double dataLoss = correctLogProbs.sumNumber().doubleValue() / samples;
            double regLoss = Transforms.pow(weightsOne, 2).sumNumber().doubleValue() * reg
                    + Transforms.pow(weightsTwo, 2).sumNumber().doubleValue() * reg;
            double loss = dataLoss + regLoss;

            if (i % 100 == 0) {
                LOG.info(String.format("iteration %d, loss %f", i, loss));
            }

            // compute the gradient on scores
            INDArray dScores = probs.dup();
            Nd4jHelper.addScalar(dScores, trainingLables, -1.0); // update correct class probabilities
            dScores.divi(samples);

            // backpropate the gradient to the parameters
            // first backprop into parameters W2 and b2
            INDArray dWeightsTwo = hiddenLayer.transpose().mmul(dScores);
            INDArray dBiasesTwo = dScores.sum(0);

            // next backprop into hidden layer
            INDArray dHiddenLayer = dScores.mmul(weightsTwo.transpose());

            // backprop the ReLU non-linearity
            dHiddenLayer.mul(hiddenLayer.gt(0)); // TODO check dhidden[hidden_layer <= 0] = 0

            // finally into W,b
            INDArray dWeightsOne = trainingData.transpose().mmul(dHiddenLayer);
            INDArray dBiasesOne = dHiddenLayer.sum(0);

            // add regularization gradient contribution
            dWeightsTwo.add(weightsTwo.mul(2.0 * reg));
            dWeightsOne.add(weightsOne.mul(2.0 * reg));

            // perform a parameter update
            weightsOne.add(dWeightsOne.mul(-1.0 * learningRate));
            biasesOne.add(dBiasesOne.mul(-1.0 * learningRate));
            weightsTwo.add(dWeightsTwo.mul(-1.0 * learningRate));
            biasesTwo.add(dBiasesTwo.mul(-1.0 * learningRate));
        }


    }

    /**
     * Use the trained weights of this two-layer network to predict labels for data points.
     *
     * @param dataSet - array of shape N x D containing N elements (of dimensionality D) to classify.
     * @return - array of shape N x 1 giving predicted labels for each input element.
     */
    public INDArray predict(INDArray dataSet) {

        INDArray hiddenLayer = dataSet.mmul(weightsOne).add(biasesOne);
        INDArray scores = hiddenLayer.mmul(weightsTwo).add(biasesTwo);

        return scores.argMax(1);
    }

}
