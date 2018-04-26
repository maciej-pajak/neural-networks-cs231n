package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.activation.ActivationFunction;
import pl.maciejpajak.network.loss.ILossFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Simple fully connected neural network.
 */
public class SimpleNetwork {

    private static final Logger logger = LoggerFactory.getLogger(SimpleNetwork.class);

    private int loggingRate = 100;
    private boolean trained = false;
    private final List<Layer> layers;
    private final ILossFunction lossFunction;

    private SimpleNetwork(List<Layer> layers, ILossFunction lossFunction) {
        this.layers = layers;
        this.lossFunction = lossFunction;
    }

    /**
     * Performs training of this neural network with provided training data and parameters.
     * @param trainingSet   data set of training examples.
     * @param validationSet data set of validation examples.
     * @param learningRate  learning rate.
     * @param reg           regularization strength.
     * @param iterations    number of iterations.
     * @param batchSize     batch size for SGD.
     * @return              learning history for this network.
     */
    public LearningHistory train(DataSet trainingSet, DataSet validationSet,
                                 double learningRate, double reg, int iterations, int batchSize) {
        LearningHistory history = new LearningHistory(iterations / loggingRate + 1);

        int samples = trainingSet.getSize();

        // Run SGD to optimize the parameters
        INDArray batchSet;
        INDArray batchLabels;
        int[] randomIndexes;

        for (int i = 1 ; i <= iterations ; i++) {
            randomIndexes = createRandomArray(samples, batchSize);
            batchSet = Nd4j.hstack(trainingSet.getData().getRows(randomIndexes), Nd4j.ones(batchSize, 1)); // vstack bias trick
            batchLabels = trainingSet.getLabels().getRows(randomIndexes);

            INDArray layerResult = batchSet;
            for (int l = 0 ; l < layers.size() ; l++) {
                layerResult = layers.get(l).forwardPass(layerResult);
            }
            // TODO
            double loss = lossFunction(layerResult, trainingSet.getLabels());

            // gradient
            INDArray layerGradient = null; // = 1????
            for (int l = layers.size() - 1 ; l >= 0 ; l++) {
                layerGradient = layers.get(l).backprop(layerGradient);
            }

            if (i % loggingRate == 0) {
                // TODO check validation accuracy and add data to history;
            }
        }

        this.trained = true;
        return history;
    }

    /**
     * Predicts labels for input data set. Can be only invoked after the network is trained.
     * @param dataSet   data set to predict classes.
     * @return          column vector with predicted labels.
     */
    public INDArray predict(DataSet dataSet) {
        if (!trained) {
            throw new IllegalStateException("Network is not trained yet.");
        }

        INDArray layerResult = dataSet.getData();
        for (int l = 0 ; l < layers.size() ; l++) {
            layerResult = layers.get(l).forwardPass(layerResult);
        }
        return layerResult;
    }

    // Getters & Setters ===========================================================================
    public void setLoggingRate(int loggingRate) {
        this.loggingRate = loggingRate;
    }

    // Layer =======================================================================================

    /**
     * Class representing single layer in network.
     */
    private static class Layer {
        private final INDArray weights;
        private final ActivationFunction activationFunction;

        private Layer(int inputSize, int outputSize, ActivationFunction activationFunction) {
            this.weights = Nd4j.randn(inputSize, outputSize);
            this.activationFunction = activationFunction;
        }

        public INDArray forwardPass(INDArray input) {
            INDArray scores = input.mmul(weights);
            return activationFunction.process(scores);
        }

        public INDArray backprop(INDArray previousGradient) {
//            return previousGradient.
        return null;
        }
    }

    // Builder =====================================================================================
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Convenient builder to create new simple network.
     */
    public static class Builder {

        private List<Layer> layers;
        private ILossFunction lossFunction;

        private Builder() {
            this.layers = new ArrayList<>();
        }

        public Builder layer(int inputSize, int outputSize, ActivationType activation) {
            layers.add(new Layer(inputSize, outputSize, activation.getFunction()));
            return this;
        }

        public Builder loss(ILossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public SimpleNetwork build() {
            if (layers.isEmpty())
                throw new IllegalStateException("At least one defined layer is required.");
            if (lossFunction == null)
                throw new IllegalStateException("Loss function cannot be null.");
            return new SimpleNetwork(layers, lossFunction);
        }

    }

    // Util ========================================================================================
    /**
     * Creates random int array without repetitions from 0 (inclusive) to upperBound (exclusive).
     * @param upperBound    upper bound of random integers.
     * @param arraySize     array size.
     * @return              random array without repetitions.
     */
    private static int[] createRandomArray(int upperBound, int arraySize) {
        return ThreadLocalRandom.current().ints(0, upperBound).distinct().limit(arraySize).toArray();
    }
}
