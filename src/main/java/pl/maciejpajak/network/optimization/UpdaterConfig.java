package pl.maciejpajak.network.optimization;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class UpdaterConfig {

    private double learningRate;
    private double momentum;

//    public UpdaterConfig(double learningRate, double momentum) {
//        this.learningRate = learningRate;
//        this.momentum = momentum;
//    }
//
//    public double getLearningRate() {
//        return learningRate;
//    }
//
//    public double getMomentum() {
//        return momentum;
//    }
}
