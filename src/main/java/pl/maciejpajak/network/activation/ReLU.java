package pl.maciejpajak.network.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Naive implementation of ReLU activation function.
 */
public class ReLU implements ActivationFunction {

    private static final Logger logger = LoggerFactory.getLogger(ReLU.class);

    private INDArray inputTmp;

    @Override
    public INDArray process(INDArray input) {
        logger.debug("input shape    : {}", Arrays.toString(input.shape()));
        this.inputTmp = input;
        return Transforms.max(input, 0);
    }

    @Override
    public INDArray backprop(INDArray input, INDArray gradient) {
        logger.debug("inputTmp shape    : {}", Arrays.toString(inputTmp.shape()));
        logger.debug("gradient shape : {}", Arrays.toString(gradient.shape()));
        return gradient.mul(inputTmp.gt(0));
    }
}
