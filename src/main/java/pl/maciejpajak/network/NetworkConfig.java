package pl.maciejpajak.network;

/**
 * Class representing simple configuration of network model.
 */
public class NetworkConfig {

    private final double regularization;
    private final double learningRate;
    private final int iterations;
    private final int batchSize;

    public NetworkConfig(double regularization, double learningRate, int iterations, int batchSize) {
        this.regularization = regularization;
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.batchSize = batchSize;
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
