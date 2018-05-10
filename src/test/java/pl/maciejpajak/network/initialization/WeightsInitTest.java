package pl.maciejpajak.network.initialization;

import org.junit.Test;

import static org.junit.Assert.*;

public class WeightsInitTest {

    @Test
    public void xavierInitTest() {
        int inputSize = 2000;
        double expVar = 1.0 / inputSize;
        double resultVar = WeightsInit.XAVIER.initialize(new int[]{inputSize,20}).varNumber().doubleValue();
        assertEquals(expVar, resultVar, 0.0001);
    }

    @Test
    public void xavierReluInitTest() {
        int inputSize = 2000;
        double expVar = 2.0 / inputSize;
        double resultVar = WeightsInit.XAVIER_RELU.initialize(new int[]{inputSize,20}).varNumber().doubleValue();
        assertEquals(expVar, resultVar, 0.00001);
    }

}