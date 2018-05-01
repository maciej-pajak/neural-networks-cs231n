package pl.maciejpajak.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import pl.maciejpajak.cifar.CifarDataSet;

public class SimpleDataSet implements DataSet {

    private final INDArray data;
    private final INDArray labels;
    private final int size;

    public SimpleDataSet(INDArray data, INDArray lables) {
        this.data = data;
        this.labels = lables;
        this.size = data.size(0);
    }

    @Override
    public CifarDataSet getSubSet(int begin, int end) {
        throw new UnsupportedOperationException("not implemented");
    }

    @Override
    public INDArray getData() {
        return data;
    }

    @Override
    public INDArray getLabels() {
        return labels;
    }

    @Override
    public int getSize() {
        return size;
    }
}
