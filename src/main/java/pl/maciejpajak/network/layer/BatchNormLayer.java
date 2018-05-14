package pl.maciejpajak.network.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BatchNormLayer implements Layer {

    @Override
    public INDArray forwardPass(INDArray input, boolean training) {
        return null;
    }

    @Override
    public INDArray backprop(INDArray previousGradient) {
        return null;
    }

    @Override
    public double getRegularizationLoss() {
        return 0;
    }

}
