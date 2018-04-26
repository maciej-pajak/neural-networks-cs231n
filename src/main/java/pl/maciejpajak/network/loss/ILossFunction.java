package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Simple interface of loss function.
 * Similar to org.nd4j.linalg.lossfunctions.ILossFunction, but much simpler.
 */
public interface ILossFunction {

    double calculateScore(INDArray data, INDArray labels, boolean average);
//    INDArray calculateScore(INDArray data, INDArray labels);

}
