package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.network.activation.ActivationFunction;

import java.util.Arrays;

/**
 * Class representing single layer in a network.
 */
public class Layer {

    private static final Logger logger = LoggerFactory.getLogger(Layer.class);

    private final INDArray weights;
    private final ActivationFunction activationFunction;
    private final NetworkConfig config;
    private INDArray inputTmp;
    private INDArray scoresTmp;

    public Layer(int inputSize, int outputSize, ActivationFunction function, NetworkConfig config) {
        this.weights = Nd4j.randn(inputSize, outputSize);
        this.activationFunction = function;
        this.config = config;
    }

//    public Layer(SimpleNetwork.LayerParams params) {
//        this.weights = Nd4j.randn(params.inputSize, params.outputSize);
//        this.activationFunction = params.function;
//    }

    public INDArray forwardPass(INDArray input, boolean training) {
        logger.debug("forward pass input shape   : {}", Arrays.toString(input.shape()));
        logger.debug("forward pass weights shape : {}", Arrays.toString(weights.shape()));
        INDArray scores = input.mmul(weights);
        logger.debug("forward pass scores shape  : {}", Arrays.toString(scores.shape()));
        if (training) {
            this.inputTmp = input;
            this.scoresTmp = scores;
        }
        return activationFunction.process(scores);
    }

    public INDArray backprop(INDArray previousGradient) {
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
        dWeights.addi(weights.mul(2.0 * config.getRegularization()));
        // update weights
        weights.subi(dWeights.muli(config.getLearningRate()));
        // return gradient on input
        return dInput;
    }

    public double getRegularizationLoss() {
        return Transforms.pow(weights, 2).mul(config.getRegularization()).sumNumber().doubleValue();
    }

}
