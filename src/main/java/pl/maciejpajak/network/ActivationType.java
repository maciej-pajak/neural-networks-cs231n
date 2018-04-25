package pl.maciejpajak.network;

import org.apache.commons.lang3.NotImplementedException;
import pl.maciejpajak.network.activation.ActivationFunction;
import pl.maciejpajak.network.activation.ReLU;

public enum ActivationType {

    RELU;

    public ActivationFunction getFunction() {
        switch (this) {
            case RELU:
                return new ReLU();
            default:
                throw new NotImplementedException("function not implemented");
        }
    }

}
