package pl.maciejpajak.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import pl.maciejpajak.cifar.CifarDataSet;

public interface DataSet {
    CifarDataSet getSubSet(int begin, int end);

    INDArray getData();

    INDArray getLabels();

    int getSize();
}
