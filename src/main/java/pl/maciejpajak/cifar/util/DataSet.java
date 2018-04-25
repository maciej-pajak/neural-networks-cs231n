package pl.maciejpajak.cifar.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface DataSet {
    CifarDataSet getSubSet(int begin, int end);

    INDArray getData();

    INDArray getLabels();

    int getSize();
}
