package pl.maciejpajak.classifier;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Linear classifier implementation.
 * Based on Stanford cs231n course.
 */
public class LinearClassifier {

    private final static Logger LOG = Logger.getLogger(LinearClassifier.class.getName());

    private INDArray weights;
    private LossFunction lossFunction;

    private LinearClassifier(INDArray weights, LossFunction lossFunction) {
        this.weights = weights;
        this.lossFunction = lossFunction;
    }

    /**
     * Trains new linear classifier using stochastic gradient descent.
     *
     *
     * @param trainingSet - array of shape N x D containing training data (N training samples each of dimension D)
     * @param trainingLabels - vector of length N containing training labels
     * @param learningRate - learning rate for optimization
     * @param reg - regularization strength
     * @param iterations - number iterations when optimizing
     * @param batchSize - number of training examples to use at each iteration
     * @param lossFunction - loss function to be used in calculations
     *
     * @return - trained linear classifier
     */
    public static LinearClassifier trainNewLinearClassifier(INDArray trainingSet, INDArray trainingLabels,
                                                            double learningRate, double reg, int iterations, int batchSize,
                                                            LossFunction lossFunction) {
        int samples = trainingSet.size(0);
        int sampleDimensions = trainingSet.size(1);
        int numClasses = trainingLabels.maxNumber().intValue() + 1; // assume y takes values 0...K-1 where K is number of classes

        // initialize weights
        INDArray weights = Nd4j.randn(sampleDimensions, numClasses).mul(0.001);

        // Run stochastic gradient descent to optimize W

        INDArray batchSet;
        INDArray batchLabels;
        int[] randomIndexes;

        for (int i = 1 ; i <= iterations ; i++) {

            // sample batchSet and corrresponding labels for current iteration
            randomIndexes = createRandomArray(samples, batchSize);
            batchSet = trainingSet.getRows(randomIndexes);
            batchLabels = trainingLabels.getRows(randomIndexes);

            // evaluate loss and gradient
            Pair<Double, INDArray> lossAndGradient = lossFunction.loss(batchSet, batchLabels, weights, reg);

            // perform parameter update
            weights.subi(lossAndGradient.getValue().mul(learningRate));

            // Update the weights using the gradient and the learning rate.
            if (iterations % 100 == 0) {
               LOG.log(Level.INFO, String.format("iteration %d / %d: loss %f", i, iterations, lossAndGradient.getKey()));
            }
        }

        return new LinearClassifier(weights, lossFunction);
    }

    /**
     * Predicts lables for input data based on learned weights.
     *
     * @param dataSet - array of shape N x D containing training data (N training samples each of dimension D)
     * @return - Predicted labels for the data in X. y_pred is a 1-dimensional array of length N,
     *          and each element is an integer giving the predicted class.
     */
    public INDArray predict(INDArray dataSet) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public enum LossFunction {
        SVM {
            @Override
            public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
                throw new UnsupportedOperationException("Not yet implemented");
            }
        },
        SVM_NAIVE {
            @Override
            public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
                int numClasses = weights.size(1);
                int samples = batchSet.size(0);

                INDArray dW = Nd4j.zeros(weights.shape());

                double loss = 0;
                INDArray scores;
                double correctClassScore;
                double margin;

                // compute loss and gradient
                for (int i = 0 ; i < samples ; i++) {
                    scores = batchSet.getRow(i).mmul(weights);
                    correctClassScore = scores.getDouble(batchLabels.getInt(i));

                    for (int j = 0 ; j < numClasses ; j++) {
                        if (j == batchLabels.getInt(i)) continue;
                        margin = scores.getDouble(j) - correctClassScore + 1;
                        if (margin > 0) {
                            loss += margin;
                            dW.getColumn(batchLabels.getInt(j)).subi(batchSet.getRow(i).transpose());
                            dW.getColumn(j).addi(batchSet.getRow(i).transpose());
                        }
                    }
                }

                // average over all examples
                loss /= samples;
                dW.divi(samples);

                // add regularization
                loss += reg * Transforms.pow(weights, 2).sumNumber().doubleValue();
                dW.addi(weights.mul(2 * reg));

                return new Pair<>(loss, dW);
            }
        },
        SOFTMAX {
            @Override
            public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
                throw new UnsupportedOperationException("Not yet implemented");
            }
        };

        /**
         * Compute the loss function and its derivative.
         *
         * @param batchSet - array of shape N x D containing a minibatch of N data points, each point has dimension D.
         * @param batchLabels - array of shape (N,) containing labels for the minibatch.
         * @param weights - array of shape D x C containing weights.
         * @param reg - regularization strength
         *
         * @return - a pair where the key is loss and value is loss gradient with respect to weights W (shape N x D)
         */
        public abstract Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg);
    }

    private static int[] createRandomArray(int upperBound, int arraySize) {
        return ThreadLocalRandom.current().ints(0, upperBound).distinct().limit(arraySize).toArray();
    }

}
