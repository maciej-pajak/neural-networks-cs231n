package pl.maciejpajak.network.activation;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

public class ReLUTest {

    @Test
    public void process() {
        // given
        INDArray random = Nd4j.randn(10,20);
        ActivationReLU reluNd4j = new ActivationReLU();
        INDArray expected = reluNd4j.getActivation(random.dup(), false);

        // when
        ReLU relu = new ReLU();
        INDArray result = relu.process(random);

        // then
        assertTrue(expected.equals(result));
    }

    @Test
    public void backprop() {
        // given
        INDArray random = Nd4j.randn(10,20);
        INDArray grad = Nd4j.randn(10,20);
        ActivationReLU reluNd4j = new ActivationReLU();
        INDArray expected = reluNd4j.backprop(random.dup(), grad).getKey();

        // when
        ReLU relu = new ReLU();
        relu.process(random); // required to store tmp
        INDArray result = relu.backprop(random, grad);

        // then
        assertTrue(expected.equals(result));
    }
}