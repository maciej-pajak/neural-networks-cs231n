package pl.maciejpajak.network.optimization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.maciejpajak.network.NetworkConfig;

public class MomentumUpdate implements Updater {

    private final NetworkConfig config;
    private final INDArray velocity;

    public MomentumUpdate(NetworkConfig config) {
        this.config = config;
        this.velocity = Nd4j.zeros(10, 10); // FIXME
    }

    @Override
    public void update(INDArray x, INDArray dx) {
        // integrate velocity: v = mu * v - learning_rate * dx
        velocity.muli(2.3).subi(dx.mul(config.getLearningRate()));
        // integrate position: x += v
        x.addi(velocity);
    }

}
