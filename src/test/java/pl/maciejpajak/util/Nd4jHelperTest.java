package pl.maciejpajak.util;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.maciejpajak.util.Nd4jHelper;

import static org.junit.Assert.*;

public class Nd4jHelperTest {

    @Test
    public void getSpecifiedElements() {
        INDArray array = Nd4j.linspace(0, 11, 12).reshape(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {3, 1});
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
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {3, 1});
        INDArray result = Nd4jHelper.getSpecifiedElementsLoop(array, columnIndexes);
        assertEquals(Nd4j.create(new double[] {2, 7, 9}, new int[]{3, 1}), result);
        assertArrayEquals(new int[]{3, 1}, result.shape());
    }

    @Test
    public void addScalar() {
        INDArray array = Nd4j.ones(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {3, 1});
        Nd4jHelper.addScalar(array, columnIndexes, -1.0);

        INDArray expected = Nd4j.create(new double[] {1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1}, new int[] {3, 4});

        assertEquals(expected, array);
        assertArrayEquals(expected.shape(), array.shape());
    }

    @Test
    public void putValuesTest() {
        INDArray array = Nd4j.linspace(0, 11, 12).reshape(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {3, 1});
        INDArray valuesToPut = Nd4j.create(new double[] {14, 15, 16}, new int[] {3, 1});

        INDArray expected = Nd4j.create(new double[] {0, 1, 14, 3, 4, 5, 6, 15, 8, 16, 10, 11}, new int[] {3, 4});
        Nd4jHelper.putValues(array, columnIndexes, valuesToPut);

        assertTrue(expected.equals(array));
    }

    @Test
    public void putScalarTest() {
        INDArray array = Nd4j.linspace(0, 11, 12).reshape(3,4);
        INDArray columnIndexes = Nd4j.create(new double[] {2, 3, 1}, new int[] {3, 1});
        INDArray valuesToPut = Nd4j.ones(3,1);

        INDArray expected = Nd4j.create(new double[] {0, 1, 1, 3, 4, 5, 6, 1, 8, 1, 10, 11}, new int[] {3, 4});
        Nd4jHelper.putScalar(array, columnIndexes, 1);

        assertTrue(expected.equals(array));
    }
}