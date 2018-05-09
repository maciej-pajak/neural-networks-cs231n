package pl.maciejpajak.network.optimization;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Updater {

    void initialize(INDArray x);
    void update(INDArray x, INDArray dx);

}
