package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.network.activation.ActivationFunction;
import pl.maciejpajak.network.initialization.WeightsInit;
import pl.maciejpajak.network.optimization.Updater;

/**
 * Class representing single layer in a network.
 */
public class Layer {

    private static final Logger logger = LoggerFactory.getLogger(Layer.class);

    private final INDArray weights;
    private final ActivationFunction activationFunction;
    private final NetworkConfig config;
    private final Updater updater;

    private INDArray inputTmp;
    private INDArray scoresTmp;

    public Layer(int inputSize, int outputSize, ActivationFunction function, NetworkConfig config, WeightsInit weightsInit) {
        this.weights = weightsInit.initialize(new int[]{inputSize, outputSize});
        this.activationFunction = function;
        this.config = config;
        this.updater = config.getUpdater();
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
        // backprop activation function
        INDArray dActivation = activationFunction.backprop(scoresTmp, previousGradient);

        // backprop dot
        INDArray dWeights = inputTmp.transpose().mmul(dActivation);
        INDArray dInput = dActivation.mmul(weights.transpose());
        // add regularization contribution
        dWeights.divi(inputTmp.size(0));
        dWeights.addi(weights.mul(2.0 * config.getRegularization()));
        // update weights
        updater.update(weights, dWeights);
//        weights.subi(dWeights.muli(config.getLearningRate()));
        // return gradient on input
        return dInput;
    }

    public double getRegularizationLoss() {
        return Transforms.pow(weights, 2).sumNumber().doubleValue() * config.getRegularization();
    }

}
