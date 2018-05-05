package pl.maciejpajak.casestudy;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.SimpleNetwork;
import pl.maciejpajak.network.activation.Identity;
import pl.maciejpajak.network.activation.ReLU;
import pl.maciejpajak.network.loss.MulticlassSVMLoss;
import pl.maciejpajak.util.Nd4jHelper;
import pl.maciejpajak.util.SimpleDataSet;


/**
 * A model of two layer fully-connected neural network.
 * Network training runs with a softmax loss function and L2 regularization.
 * The network uses a ReLU nonlinearity after the first fully connected layer
 *
 * Architecture:
 * input - fully connected layer - ReLU - fully connected layer - softmax
 *
 * The outputs of the second fully-connected layer are the scores for each class.
 *
 * Based on cs231n.
 */
public class TwoLayerNetworkCaseStudy {

    private final static Logger LOG = LoggerFactory.getLogger(TwoLayerNetworkCaseStudy.class.getName());

    // first layer weights D x H
    private final INDArray weightsOne;
    // first layer biases 1 x H
    private final INDArray biasesOne;

    // second layer weights H x C
    private final INDArray weightsTwo;
    // second layer biases 1 x C
    private final INDArray biasesTwo;

    private INDArray learningHistory;

    /**
     * Initialize the model. Weights are initialized to small random values and
     * biases are initialized to zero.
     *
     * @param inputSize - the dimension D of input data.
     * @param hiddenSize - the number of neurons H in the hidden layer.
     * @param outputSize - the number of classes.
     * @param std - random weights initialization coefficient.
     */
    public TwoLayerNetworkCaseStudy(int inputSize, int hiddenSize, int outputSize, double std) {
        weightsOne = Nd4j.randn(inputSize, hiddenSize).mul(std);
        biasesOne = Nd4j.zeros(1, hiddenSize);
        weightsTwo = Nd4j.randn(hiddenSize, outputSize).mul(std);
        biasesTwo = Nd4j.zeros(1, outputSize);
    }

    /**
     * Train this neural network using stochastic gradient descent.
     *
     * @param trainingData - an array of shape N x D giving training data.
     * @param trainingLables - an array of shape N x 1 giving training labels.
     * @param learningRate - rate of optimization.
     * @param reg - regularization strength.
     * @param iterations - number of iterations.
     */
    public void train(INDArray trainingData, INDArray trainingLables, double learningRate, double reg, int iterations) {

        int samples = trainingData.size(0);
        learningHistory = Nd4j.create(iterations / 100, 3);

        for (int i = 1 ; i <= iterations ; i++) {
            // evaluate class scores N x C
            INDArray hiddenLayer = Transforms.max(trainingData.mmul(weightsOne).addRowVector(biasesOne), 0); // ReLU activation
            INDArray scores = hiddenLayer.mmul(weightsTwo).addRowVector(biasesTwo);

            // compute the class probabilities
            INDArray expScores = Transforms.exp(scores);
            INDArray probs = expScores.divColumnVector(expScores.sum(1));

            // compute the loss: average cross-entropy loss and regularization
            INDArray correctLogProbs = Transforms.log(Nd4jHelper.getSpecifiedElements(probs, trainingLables)).mul(-1);

            double dataLoss = correctLogProbs.sumNumber().doubleValue() / samples;
            double regLoss = Transforms.pow(weightsOne, 2).sumNumber().doubleValue() * reg
                    + Transforms.pow(weightsTwo, 2).sumNumber().doubleValue() * reg;
            double loss = dataLoss + regLoss;

            if (i % 100 == 0) {
                INDArray predicted = this.predict(trainingData);
                double acc = predicted.eq(trainingLables).sumNumber().doubleValue() / trainingLables.length();
                learningHistory.putRow(i / 100 - 1, Nd4j.create(new double[]{i, acc, loss}, new int[]{1,3}));
                LOG.info(String.format("iteration %d / %d, acc %f, loss %f", i, iterations, acc, loss));
            }

            // compute the gradient on scores
            INDArray dScores = probs.dup();
            Nd4jHelper.addScalar(dScores, trainingLables, -1.0); // update correct class probabilities
            dScores.divi(samples);

            // backpropate the gradient to the parameters
            // first backprop into parameters W2 and b2
            INDArray dWeightsTwo = hiddenLayer.transpose().mmul(dScores);
            INDArray dBiasesTwo = dScores.sum(0);

            // next backprop into hidden layer
            INDArray dHiddenLayer = dScores.mmul(weightsTwo.transpose());

            // backprop the ReLU non-linearity
            dHiddenLayer.muli(hiddenLayer.gt(0)); // TODO check dhidden[hidden_layer <= 0] = 0

            // finally into W,b
            INDArray dWeightsOne = trainingData.transpose().mmul(dHiddenLayer);
            INDArray dBiasesOne = dHiddenLayer.sum(0);

            // add regularization gradient contribution
            dWeightsTwo.addi(weightsTwo.mul(2.0 * reg));
            dWeightsOne.addi(weightsOne.mul(2.0 * reg));

            // perform a parameter update
            weightsOne.addi(dWeightsOne.mul(-1.0 * learningRate));
            biasesOne.addi(dBiasesOne.mul(-1.0 * learningRate));
            weightsTwo.addi(dWeightsTwo.mul(-1.0 * learningRate));
            biasesTwo.addi(dBiasesTwo.mul(-1.0 * learningRate));
        }


    }

    /**
     * Use the trained weights of this two-layer network to predict labels for data points.
     *
     * @param dataSet - array of shape N x D containing N elements (of dimensionality D) to classify.
     * @return - array of shape N x 1 giving predicted labels for each input element.
     */
    public INDArray predict(INDArray dataSet) {

        INDArray hiddenLayer = Transforms.max(dataSet.mmul(weightsOne).addRowVector(biasesOne), 0);
        INDArray scores = hiddenLayer.mmul(weightsTwo).addRowVector(biasesTwo);

        return scores.argMax(1);
    }

    public void printLearningAnalysis() {
        // Create Chart
        XYChart chart = new XYChartBuilder().width(600).height(500).title("Learning analysis").xAxisTitle("iteration").yAxisTitle("val").build();

        double[] xIterations = learningHistory.getColumn(0).dup().data().asDouble();
        double[] yAccuracy = learningHistory.getColumn(1).dup().data().asDouble();
        double[] yLoss = learningHistory.getColumn(2).div(learningHistory.getColumn(2).maxNumber()).dup().data().asDouble(); // rescale loss to percentage

        // Series
        chart.addSeries("Accuracy [%]", xIterations, yAccuracy);
        chart.addSeries("Loss [%  of max loss]", xIterations, yLoss);

        new SwingWrapper(chart).displayChart();
    }

    public static void main(String[] args) {
        // generate data that is not lineary separable
        final int pointsPreClass = 100;
        final int dimensionality = 2;
        final int numberOfClasses = 3;

        INDArray dataSet = Nd4j.zeros(pointsPreClass * numberOfClasses, dimensionality);
        INDArray dataLabels = Nd4j.zeros(pointsPreClass * numberOfClasses, 1);

        for (int i = 0 ; i < numberOfClasses ; i++) {
            // INDArray ix = Nd4j.linspace(pointsPreClass * i, pointsPreClass * (i + 1) - 1, pointsPreClass);
            INDArray radius = Nd4j.linspace(0.0, 1,pointsPreClass);
            INDArray theta = Nd4j
                    .linspace(i * 4, (i + 1) * 4, pointsPreClass)
                    .addi(Nd4j.randn(1, pointsPreClass).muli(0.2));
            INDArray coord = Nd4j.vstack(radius.mul(Transforms.sin(theta)), radius.mul(Transforms.cos(theta))).transpose();

            for (int j = 0 ; j < pointsPreClass ; j++) { // TODO refactor to utilize Nd4j without loop
                dataSet.putRow(i * pointsPreClass + j, coord.getRow(j));
                dataLabels.putScalar(i * pointsPreClass + j, i);
            }
        }

//        TwoLayerNetworkCaseStudy network =
//                new TwoLayerNetworkCaseStudy(dataSet.size(1), 100, 3, 0.01);
//
//        network.train(dataSet, dataLabels,0.5, 0.001,10000);
//
//        INDArray predicted = network.predict(dataSet);
//        double acc = predicted.eq(dataLabels).sumNumber().doubleValue() / dataLabels.length();
//
//        LOG.info("final accuracy = {}", acc);

//        network.printLearningAnalysis();

        SimpleNetwork simpleNetwork = SimpleNetwork.builder()
                .layer(dataSet.size(1), 100, new ReLU())
                .layer(100, 3, new Identity())
                .loss(new MulticlassSVMLoss())
                .learningRate(0.5)
                .learningRateDecay(1.0)
                .regularization(0.001)
                .iterations(10000)
                .batchSize(100)
                .build();
        LearningHistory lh = simpleNetwork.train(new SimpleDataSet(dataSet, dataLabels), null);
        LOG.info("simple network accuracy: {}", simpleNetwork.predict(dataSet).eq(dataLabels).meanNumber().doubleValue());
    }

}
