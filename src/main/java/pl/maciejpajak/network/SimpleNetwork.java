package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossHinge;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.cifar.util.DataSet;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.activation.ActivationFunction;

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
    private List<Layer> layers;
    private SimpleNetwork(List<Layer> layers) {

    }

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
            // loss = lossFunction(layerResult, labels);

            // gradient
            INDArray layerGradient = null; // = 1????
            for (int l = layers.size() - 1 ; l >= 0 ; l++) {
                layerGradient = layers.get(l).backprop(layerGradient);
            }

            if (i % loggingRate == 0) {
                // TODO check validation accuracy and data to history;
            }
        }

        this.trained = true;
        return history;
    }

    // Getters & Setters ===========================================================================
    public void setLoggingRate(int loggingRate) {
        this.loggingRate = loggingRate;
    }

    // Layer =======================================================================================
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

    public static class Builder {

        private List<Layer> layers;

        private Builder() {
            this.layers = new ArrayList<>();
        }

        public Builder layer(int inputSize, int outputSize, ActivationType activation) {
            layers.add(new Layer(inputSize, outputSize, activation.getFunction()));
            return this;
        }

        public SimpleNetwork build() {
            if (layers.isEmpty()) throw new IllegalStateException("At least one defined layer is required.");
            return new SimpleNetwork(layers);
        }

    }

    // Util ========================================================================================
    private static int[] createRandomArray(int upperBound, int arraySize) {
        return ThreadLocalRandom.current().ints(0, upperBound).distinct().limit(arraySize).toArray();
    }
}
