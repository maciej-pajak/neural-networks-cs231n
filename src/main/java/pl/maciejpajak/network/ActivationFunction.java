package pl.maciejpajak.network;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActivationFunction {

    public INDArray process(INDArray input);
    public INDArray backprop(INDArray input);

}
