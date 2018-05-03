package pl.maciejpajak.network.loss;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SoftmaxLossTest {

    private INDArray scores;
    private INDArray lables;
    private INDArray binaryLabels;
    private SoftmaxLoss softmax;

    @Before
    public void setUp() throws Exception {
        scores = Nd4j.create(new double[] {
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0
        }, new int[] {5, 4});
        binaryLabels = Nd4j.create(new double[] {
                1, 0, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                0, 1, 0, 0,
                0, 0, 1, 0,
        }, new int[] {5, 4});
        lables = Nd4j.create(new double[] {0, 2, 3, 1, 2}, new int[] {5, 1});
        softmax = new SoftmaxLoss();
    }

    @Test
    public void calculateScore() {
        double calcScore = softmax.calculateScore(scores, lables, false);
        double calcAvgScore = softmax.calculateScore(scores, lables, true);

        double checkWithNd4jScore = new LossMCXENT().computeScore(binaryLabels, scores, new ActivationSoftmax(), null, true);

        assertEquals(6.9126, calcScore, 0.0001);
        assertEquals(1.3825, calcAvgScore, 0.0001);
        assertEquals(checkWithNd4jScore, calcAvgScore, 0.0001);
    }

    @Test
    public void calculateGradient() {
        INDArray gradientExpected = Nd4j.create(new double[]{
                -0.78616178, 0.236327782, 0.261182592, 0.288651405,
                0.21383822, 0.236327782, -0.738817408, 0.288651405,
                0.21383822, 0.236327782, 0.261182592, -0.711348595,
                0.21383822, -0.763672218, 0.261182592, 0.288651405,
                0.21383822, 0.236327782, -0.738817408, 0.288651405
        }, new int[]{5, 4});
        softmax.calculateScore(scores, lables, false); // forward pass is required for loss function to cache values
        INDArray result = softmax.calculateGradient(lables);
        assertTrue(gradientExpected.equalsWithEps(result, 0.001));
        assertTrue(new LossMCXENT().computeGradient(binaryLabels, scores, new ActivationSoftmax(), null)
                .equalsWithEps(result, 0.001));

    }

}