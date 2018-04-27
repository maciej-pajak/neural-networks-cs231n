package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.activation.ActivationFunction;
import pl.maciejpajak.network.activation.ReLU;
import pl.maciejpajak.network.loss.ILossFunction;
import pl.maciejpajak.util.DataSet;

import java.util.ArrayList;
import java.util.Arrays;
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

    // maybe this should be solved differently?
    private double regularization;
    private double learningRate;

    private SimpleNetwork(List<LayerParams> layers, ILossFunction lossFunction) {
        this.layers = new ArrayList<>(layers.size());
        for (int i = 0 ; i < layers.size() ; i++) {
            this.layers.add(new Layer(layers.get(i)));
        }
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
        this.regularization = reg;
        this.learningRate = learningRate;
        LearningHistory history = new LearningHistory(iterations / loggingRate + 1);
        int samples = trainingSet.getSize();

        // Run SGD to optimize the parameters
        INDArray batchSet;
        INDArray batchLabels;
        int[] randomIndexes;

        for (int i = 1 ; i <= iterations ; i++) {
            randomIndexes = createRandomArray(samples, batchSize);
            batchSet = trainingSet.getData().getRows(randomIndexes); // vstack bias trick
            batchLabels = trainingSet.getLabels().getRows(randomIndexes);

            INDArray layerResult = batchSet;
            for (int l = 0 ; l < layers.size() ; l++) {
                logger.debug("forward pass layer : {}", l);
                layerResult = Nd4j.hstack(layerResult, Nd4j.ones(layerResult.size(0), 1));
                layerResult = layers.get(l).forwardPass(layerResult, true);
            }
            logger.debug("batchLabels shape : {}", Arrays.toString(batchLabels.shape()));
            logger.debug("layerResult shape : {}", Arrays.toString(layerResult.shape()));
            double loss = lossFunction.calculateScore(layerResult, batchLabels, true);

            // gradient ==========
            INDArray layerGradient = lossFunction.calculateGradient(batchLabels);
            logger.debug("loss gradient shape {} : ", Arrays.toString(layerGradient.shape()));
            for (int l = layers.size() - 1 ; l >= 0 ; l--) {
                layerGradient = layers.get(l).backprop(layerGradient);
            }

            if (i % loggingRate == 0) {
                // TODO add data to history
                double valAcc = predictLabels(validationSet.getData()).eq(validationSet.getLabels()).meanNumber().doubleValue();
                history.addNextRecord(i, loss, valAcc);
                logger.info("val_acc = {}, loss = {} ({} / {})", valAcc, loss, i / iterations);
            }
        }

        this.trained = true;
        return history;
    }

    /**
     * Predicts labels for input data set. Can be only invoked after the network was trained.
     * @param data   data set to predict classes.
     * @return          column vector with predicted labels.
     */
    public INDArray predict(INDArray data) {
        if (!trained) {
            throw new IllegalStateException("Network is not trained yet.");
        }
        return predictLabels(data);
    }

    private INDArray predictLabels(INDArray data) {
        INDArray layerResult = data;
        for (int l = 0 ; l < layers.size() ; l++) {
            layerResult = Nd4j.hstack(layerResult, Nd4j.ones(layerResult.size(0), 1));
            layerResult = layers.get(l).forwardPass(layerResult, false);
        }
        return layerResult.argMax(1);
    }

    // Getters & Setters ===========================================================================
    public void setLoggingRate(int loggingRate) {
        this.loggingRate = loggingRate;
    }

    // Layer =======================================================================================

    /**
     * Class representing single layer in a network.
     */
    private class Layer {

        private final INDArray weights;
        private final ActivationFunction activationFunction;
        private INDArray inputTmp;
        private INDArray scoresTmp;

        private Layer(LayerParams params) {
            this.weights = Nd4j.randn(params.inputSize, params.outputSize);
            this.activationFunction = params.function;
        }

        public INDArray forwardPass(INDArray input, boolean training) {
            INDArray scores = input.mmul(weights);
            if (training) {
                this.inputTmp = input;
                this.scoresTmp = scores;
            }
            return activationFunction.process(scores);
        }

        public INDArray backprop(INDArray previousGradient) {
            logger.debug("backproping layer");
            logger.debug("previous gradient shape : {}", Arrays.toString(previousGradient.shape()));
            // backprop activation function
            INDArray dActivation = activationFunction.backprop(scoresTmp, previousGradient);
            logger.debug("dActivation shape       : {}", Arrays.toString(dActivation.shape()));
            logger.debug("inputTmp shape          : {}", Arrays.toString(inputTmp.shape()));
            logger.debug("weights shape           : {}", Arrays.toString(weights.shape()));
            // backprop dot
            INDArray dWeights = inputTmp.transpose().mmul(dActivation);
            INDArray dInput = dActivation.mmul(weights.transpose());
            // add regularization contribution
            dWeights.addi(weights.mul(2.0 * regularization));
            // update weights
            weights.subi(dWeights.muli(learningRate));
            // return gradient on input
            return dInput;
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

        private List<LayerParams> layers;
        private ILossFunction lossFunction;

        private Builder() {
            this.layers = new ArrayList<>();
        }

        public Builder layer(int inputSize, int outputSize, ActivationFunction activation) {
            layers.add(new LayerParams(inputSize + 1, outputSize, activation)); // +1 for bias
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

    /**
     * Data class used to store single layer parameters.
     */
    private static class LayerParams { // is this acceptable solution?
        private int inputSize;
        private int outputSize;
        private ActivationFunction function;
        private LayerParams(int inputSize, int outputSize, ActivationFunction function) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.function = function;
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
