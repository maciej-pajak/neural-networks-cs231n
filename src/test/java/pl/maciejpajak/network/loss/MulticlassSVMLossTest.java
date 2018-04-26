package pl.maciejpajak.network.loss;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class MulticlassSVMLossTest {

    @Test
    public void calculateScore() {
        INDArray scores = Nd4j.create(new double[] {
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0
        }, new int[] {5, 4});
        INDArray lables = Nd4j.create(new double[] {0, 2, 3, 1, 2}, new int[] {5, 1});

        MulticlassSVMLoss svm = new MulticlassSVMLoss();
        assertEquals(2.96, svm.calculateScore(scores, lables, true), 0.0001);
    }
}