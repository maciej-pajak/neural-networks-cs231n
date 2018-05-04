package pl.maciejpajak.network;

/**
 * Class representing simple configuration of network model.
 */
public class NetworkConfig {

    private final double regularization;
    private final double learningRateDecay;
    private double learningRate;
    private final int iterations;
    private final int batchSize;

    public NetworkConfig(double regularization, double learningRate, double learningRateDecay, int iterations, int batchSize) {
        this.regularization = regularization;
        this.learningRate = learningRate;
        this.learningRateDecay = learningRateDecay;
        this.iterations = iterations;
        this.batchSize = batchSize;
    }

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
}
