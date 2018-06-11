package pl.maciejpajak.network.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

    INDArray forwardPass(INDArray input, boolean training);
    INDArray backprop(INDArray previousGradient);
    double getRegularizationLoss();

}
