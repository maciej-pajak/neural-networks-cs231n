package pl.maciejpajak.cifar.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Cifar data containing images and labels.
 */
public class CifarDataSet {

    // an array of shape N x D giving N examples (each with D dimensionality)
    private final INDArray data;
    // an array of shape N x 1 giving labels for each example from data array
    private final INDArray labels;

    private final int size;

    private CifarDataSet(INDArray data, INDArray labels) {
        if (data.shape()[0] != labels.shape()[0] || !labels.isColumnVector()) {
            throw new IllegalArgumentException("labels matrix is invalid");
        }
        this.data = data;
        this.labels = labels;
        this.size = data.shape()[0];
    }

    /**
     * Returns subset of this CifarDataSet. The returned object is view of the original set.
     * Examples are indexed from 1.
     *
     * @param begin - start element index (inclusive)
     * @param end - end element index (exclusive)
     * @return - subset view of CifarDataSet.
     */
    public CifarDataSet getSubSet(int begin, int end) {
        return new CifarDataSet(data.get(NDArrayIndex.interval(begin - 1, end - 1), NDArrayIndex.all()),
                labels.get(NDArrayIndex.interval(begin - 1, end - 1), NDArrayIndex.all()));
    }

    /**
     * Returns mean example for this data set.
     * @return
     */
    public INDArray getMeanExample() {
        return data.mean(0);
    }

    /**
     * Subtracts mean from every example in this data set.
     * @param mean
     */
    public void preprocessWithMean(INDArray mean) {
        data.subiRowVector(mean);
    }

    /**
     * Loads CifarDataSet from disk.
     *
     * @param imagesInFile - number of images in one file.
     * @param imageLength - length of single image
     * @param files - files to load from.
     * @return CifarDataSet.
     */
    public static CifarDataSet loadFromDisk(int imagesInFile, int imageLength, File... files) {
        INDArray dataSet = Nd4j.create(imagesInFile * files.length, imageLength);
        INDArray labels = Nd4j.create(imagesInFile * files.length, 1);
        int imgCount = 0;
        try {
            FileInputStream fis;
            byte[] buffer = new byte[3072];
            for (File f : files) {
                fis = new FileInputStream(f);
                for (int j = 0; j < imagesInFile; j++) {
                    labels.putScalar(imgCount, fis.read());

                    fis.read(buffer);

                    for (int i = 0; i < imageLength; i++) {
                        dataSet.putScalar(imgCount, i, buffer[i]);
                    }
                    imgCount++;

                }
                fis.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new CifarDataSet(dataSet, labels);
    }

    public INDArray getData() {
        return data;
    }

    public INDArray getLabels() {
        return labels;
    }

    public int getSize() {
        return size;
    }
}
