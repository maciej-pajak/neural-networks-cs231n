package pl.maciejpajak.network.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Identity implements ActivationFunction {

    @Override
    public INDArray process(INDArray input) {
        return input;
    }

    @Override
    public INDArray backprop(INDArray input, INDArray gradient) {
        return gradient;
    }

}
