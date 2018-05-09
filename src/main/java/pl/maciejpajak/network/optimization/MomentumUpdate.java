package pl.maciejpajak.network.optimization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MomentumUpdate implements Updater {

    private final UpdaterConfig config;
    private INDArray velocity;

    public MomentumUpdate(UpdaterConfig config) {
        this.config = config;
    }

    @Override
    public void initialize(INDArray x) {
        this.velocity = Nd4j.zeros(x.shape());
    }

    @Override
    public void update(INDArray x, INDArray dx) {
        // integrate velocity: v = mu * v - learning_rate * dx
        velocity.muli(config.getMomentum()).subi(dx.mul(config.getLearningRate()));
        // integrate position: x += v
        x.addi(velocity);
    }

}
