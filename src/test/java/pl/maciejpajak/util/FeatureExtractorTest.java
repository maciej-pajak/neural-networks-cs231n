package pl.maciejpajak.util;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class FeatureExtractorTest {

    @Test
    public void extractFeaturesGreyScale() {
        // given
        INDArray lables = Nd4j.rand(2,1);
        DataSet set = new SimpleDataSet(Nd4j.linspace(0, 110, 12).reshape(2, 6), lables);
        DataSet expectedResult = new SimpleDataSet(Nd4j.create(new double[]{8.15, 38.15, 68.15, 98.15}, new int[]{2, 2}), lables);

        // when
        DataSet result = FeatureExtractor.extractFeatures(set, FeatureExtractor.Feature.GREY_SCALE);

        // then
        assertTrue(expectedResult.getData().equalsWithEps(result.getData(), 0.001));
        assertEquals(expectedResult.getLabels(), result.getLabels());
    }
}