package pl.maciejpajak.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

public class FeatureExtractor {

    public static DataSet extractFeatures(DataSet dataSet, Feature... features) {
        List<INDArray> featuresList = new ArrayList<>();
        for (Feature f : features) {
            featuresList.add(f.extract(dataSet.getData()));
        }
        return new SimpleDataSet(Nd4j.hstack(featuresList), dataSet.getLabels().dup());
    }

    public enum Feature {
        GREY_SCALE {
            @Override
            INDArray extract(INDArray data) {
                INDArray coeff = Nd4j.create(new double[] {0.299, 0.587, 0.114}, new int[] {3,1});
                INDArray result = Nd4j.create(data.rows(), data.columns() / 3);

                for (int i = 0 ; i < result.columns() ; i++) {
                    result.putColumn(i, data.get(NDArrayIndex.all(), NDArrayIndex.interval(i * 3, i * 3 + 3)).mmul(coeff));
                }

                return result;
            }
        };

        abstract INDArray extract(INDArray data);
    }
}
