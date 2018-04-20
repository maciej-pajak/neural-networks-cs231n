package pl.maciejpajak.classifier;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.SpecifiedIndex;


public final class Nd4jHelper {

    private Nd4jHelper() {}

    public static INDArray getSpecifiedElements(INDArray source, INDArray columnIndices) {
        int rowsCount = source.size(0);
        INDArray indices = Nd4j.linspace(0, rowsCount - 1, rowsCount)
                .muli(source.size(1))
                .add(columnIndices.transpose());
        return Nd4j.toFlattened(source).get(new SpecifiedIndex(indices.data().asInt()));
    }

    public static INDArray getSpecifiedElementsLoop(INDArray source, INDArray columnIndices) {
        int rowsCount = source.size(0);
        INDArray result = Nd4j.create(rowsCount, 1);
        for (int i = 0 ; i < rowsCount ; i++) {
            result.put(i, 0, source.getDouble(i, columnIndices.getInt(0, i)));
        }
        return result;
    }

    public static void putValues(INDArray destination, INDArray columnIndices, INDArray values) {
        int rowsCount = destination.size(0);
        for (int i = 0 ; i < rowsCount ; i++) {
            destination.putScalar(i, columnIndices.getInt(i, 0), values.getDouble(i, 0));
        }
    }

    public static void putScalar(INDArray destination, INDArray columnIndices, double value) {
        int rowsCount = destination.size(0);
        for (int i = 0 ; i < rowsCount ; i++) {
            destination.putScalar(i, columnIndices.getInt(i, 0), value);
        }
    }

    public static void addScalar(INDArray destination, INDArray columnIndices, double value) {
        int rowsCount = destination.size(0);
        for (int i = 0 ; i < rowsCount ; i++) {
            destination.getScalar(i, columnIndices.getInt(i, 0)).add(value);
        }
    }

}
