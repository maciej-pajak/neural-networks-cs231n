package pl.maciejpajak.network.optimization;

import org.nd4j.linalg.api.ndarray.INDArray;
import pl.maciejpajak.network.NetworkConfig;

public class Sgd implements Updater {

    private final NetworkConfig config;

    public Sgd(NetworkConfig config) {
        this.config = config;
    }

    @Override
    public void update(INDArray x, INDArray dx) {
        x.subi(dx.muli(config.getLearningRate()));
    }
}
