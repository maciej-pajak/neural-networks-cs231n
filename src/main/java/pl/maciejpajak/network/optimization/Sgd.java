package pl.maciejpajak.network.optimization;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Sgd implements Updater {

    private final UpdaterConfig config;

    public Sgd(UpdaterConfig config) {
        this.config = config;
    }

    @Override
    public void initialize(INDArray x) {
        // unused in sgd
    }

    @Override
    public void update(INDArray x, INDArray dx) {
        x.subi(dx.muli(config.getLearningRate()));
    }
}
