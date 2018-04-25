package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class SimpleNetwork {

    private List<Layer> layers;

    private SimpleNetwork(List<Layer> layers) {

    }


    // Layer =======================================================================================
    private static class Layer {
        private final INDArray weights;
        private final ActivationFunction activationFunction;

        private Layer(int inputSize, int outputSize, ActivationFunction activationFunction) {
            this.weights = Nd4j.randn(inputSize, outputSize);
            this.activationFunction = activationFunction;
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
}
