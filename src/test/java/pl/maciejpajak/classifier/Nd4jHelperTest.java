package pl.maciejpajak.classifier;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class Nd4jHelperTest {

    @Test
    public void getSpecifiedElements() {
        INDArray array = Nd4j.linspace(0, 11, 12).reshape(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {1, 3});
        // 0 1 2 3
        // 4 5 6 7
        // 8 9 10 11
        INDArray result = Nd4jHelper.getSpecifiedElements(array, columnIndexes);
        assertEquals(Nd4j.create(new double[] {2, 7, 9}, new int[]{3, 1}), result);
        assertArrayEquals(new int[]{3, 1}, result.shape());
    }


    @Test
    public void getSpecifiedElementsLoop() {
        INDArray array = Nd4j.linspace(0, 11, 12).reshape(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {1, 3});
        INDArray result = Nd4jHelper.getSpecifiedElementsLoop(array, columnIndexes);
        assertEquals(Nd4j.create(new double[] {2, 7, 9}, new int[]{3, 1}), result);
        assertArrayEquals(new int[]{3, 1}, result.shape());
    }

}