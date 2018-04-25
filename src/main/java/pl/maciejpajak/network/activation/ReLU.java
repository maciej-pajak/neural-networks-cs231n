package pl.maciejpajak.network.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Naive implementation of ReLU activation function.
 */
public class ReLU implements ActivationFunction {

    @Override
    public INDArray process(INDArray input) {
        return Transforms.max(input, 0);
    }

    @Override
    public INDArray backprop(INDArray input, INDArray gradient) {
        return gradient.mul(input.gt(0));
    }
}
