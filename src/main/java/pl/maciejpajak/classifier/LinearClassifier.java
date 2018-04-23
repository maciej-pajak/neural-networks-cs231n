package pl.maciejpajak.classifier;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple linear classifier implementation.
 * Based on Stanford cs231n course.
 */
public class LinearClassifier {

    private final static Logger LOG = Logger.getLogger(LinearClassifier.class.getName());

    private final INDArray weights;
    private final LossFunction lossFunction;

    private final LearningHistory learningHistory;

//    private LinearClassifier(INDArray weights, LossFunction lossFunction) {
//        this(weights, lossFunction, null);
//    }

    private LinearClassifier(INDArray weights, LossFunction lossFunction, LearningHistory learningHistory) {
        this.weights = weights;
        this.lossFunction = lossFunction;
        this.learningHistory = learningHistory;
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
        final int loggingRate = 100;
        LearningHistory learningHistory = new LearningHistory(iterations / loggingRate); // for learning analysis

        int samples = trainingSet.size(0);
        int sampleDimensions = trainingSet.size(1);
        int numClasses = trainingLabels.maxNumber().intValue() + 1; // assume y takes values 0...K-1 where K is number of classes

        // initialize weights
        INDArray weights = Nd4j.randn(sampleDimensions + 1, numClasses).mul(0.0001); // + 1 bias trick

        // Run stochastic gradient descent to optimize W

        INDArray batchSet;
        INDArray batchLabels;
        int[] randomIndexes;

        for (int i = 1 ; i <= iterations ; i++) {

            // sample batchSet and corrresponding labels for current iteration
            randomIndexes = createRandomArray(samples, batchSize);
            batchSet = Nd4j.hstack(trainingSet.getRows(randomIndexes), Nd4j.ones(batchSize, 1)); // vstack bias trick
            batchLabels = trainingLabels.getRows(randomIndexes);

            // evaluate loss and gradient
            Pair<Double, INDArray> lossAndGradient = lossFunction.loss(batchSet, batchLabels, weights, reg);

            // perform parameter update
            weights.subi(lossAndGradient.getValue().mul(learningRate));

            // Update the weights using the gradient and the learning rate.
            if (i % loggingRate == 0) {
                LOG.log(Level.INFO, String.format("iteration %d / %d: loss %f", i, iterations, lossAndGradient.getKey()));
                learningHistory.addNextRecord(i, lossAndGradient.getKey(), 0.0);
//                INDArray bestScore = batchSet.mmul(weights).argMax(1);
//                double acc = bestScore.eq(batchLabels).sumNumber().doubleValue() / batchSize;
//                learningHistory.putRow(i / loggingRate - 1, Nd4j.create(new double[]{i, acc, lossAndGradient.getKey()}, new int[]{1,3}));
//                LOG.log(Level.INFO, String.format("iteration %d / %d: accuracy %f ; loss %f", i, iterations, acc, lossAndGradient.getKey()));
            }
        }
        return new LinearClassifier(weights, lossFunction, learningHistory);
    }

    /**
     * Predicts lables for input data based on learned weights.
     *
     * @param dataSet - array of shape N x D containing training data (N training samples each of dimension D)
     * @return - Predicted labels for the data in X. Returns 1-dimensional array of length N,
     *          where each element is an integer giving the predicted class.
     */
    public INDArray predict(INDArray dataSet) {

        INDArray scores = Nd4j.hstack(dataSet, Nd4j.ones(dataSet.rows(), 1)).mmul(weights);

        INDArray bestScores = scores.argMax(1);

        return bestScores;
    }

    private static int[] createRandomArray(int upperBound, int arraySize) {
        return ThreadLocalRandom.current().ints(0, upperBound).distinct().limit(arraySize).toArray();
    }

    public LearningHistory getLearningHistory() {
        return learningHistory;
    }

    public INDArray getWeights() {
        return weights;
    }
}
