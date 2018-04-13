package pl.maciejpajak.playground;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class LinerClassifierPlaygroundTest {

    @Test
    public void l() {
        INDArray x = Nd4j.create(new double[]{0.1, 0.7, 1.3, 1.9, 2.5,
                0.2, 0.8 ,1.4, 2.0, 2.6,
                0.3, 0.9, 1.5, 2.1, 2.7,
                0.4, 1.0, 1.6, 2.2, 2.8}, new int[]{4, 5});
        INDArray y = Nd4j.create(new double[]{0, 1, 2, 3, 4}, new int[]{1, 5});
        INDArray W = Nd4j.create(new double[]{0.1, 0.6, 1.1, 1.6,
                0.2, 0.7, 1.2, 1.7,
                0.3, 0.8, 1.3, 1.8,
                0.4, 0.9, 1.4, 1.9,
                0.5, 1.0, 1.5, 2.0}, new int[]{5, 4});

        assertEquals(16.86, LinerClassifierPlayground.L(W, y, x), 0.01);

    }
}