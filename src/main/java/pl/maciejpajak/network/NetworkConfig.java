package pl.maciejpajak.network;

import lombok.AllArgsConstructor;
import pl.maciejpajak.network.optimization.Updater;

/**
 * Class representing simple configuration of network model.
 */
@AllArgsConstructor
public class NetworkConfig {

    private double regularization;
    private double learningRateDecay;
    private double learningRate;
    private int iterations;
    private int batchSize;
    private Updater updater;

//    public NetworkConfig(double regularization, double learningRate, double learningRateDecay, int iterations, int batchSize) {
//        this.regularization = regularization;
//        this.learningRate = learningRate;
//        this.learningRateDecay = learningRateDecay;
//        this.iterations = iterations;
//        this.batchSize = batchSize;
//    }

    public void decayLearningRate() {
        this.learningRate *= learningRateDecay;
    }

    public double getRegularization() {
        return this.regularization;
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public int getIterations() {
        return iterations;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public Updater getUpdater() {
        return updater;
    }
}
