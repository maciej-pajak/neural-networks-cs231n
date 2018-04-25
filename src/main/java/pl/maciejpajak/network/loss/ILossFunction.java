package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ILossFunction {

    INDArray calculateScore(INDArray data, INDArray labels);
//    INDArray calculateGradient(INDArray )
}
