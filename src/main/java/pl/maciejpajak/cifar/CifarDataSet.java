package pl.maciejpajak.cifar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import pl.maciejpajak.util.DataSet;

import java.awt.*;
import java.awt.image.*;

/**
 * Cifar data containing images and labels.
 */
public class CifarDataSet implements DataSet {

    private static final int IMAGE_WIDTH = 32;
    private static final int IMAGE_HEIGHT = 32;

    // an array of shape N x D giving N examples (each with D dimensionality)
    private final INDArray data;
    // an array of shape N x 1 giving labels for each example from data array
    private final INDArray labels;

    private final int size;

    public CifarDataSet(INDArray data, INDArray labels) {
        if (labels != null && (data.shape()[0] != labels.shape()[0] || !labels.isColumnVector())) {
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
    @Override
    public CifarDataSet getSubSet(int begin, int end) {
        return new CifarDataSet(data.get(NDArrayIndex.interval(begin - 1, end - 1), NDArrayIndex.all()),
                labels == null ? null : labels.get(NDArrayIndex.interval(begin - 1, end - 1), NDArrayIndex.all()));
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

    public void rescale(int scale) {
        double min = data.minNumber().doubleValue();
        double max = data.maxNumber().doubleValue();
        data.subi(min).divi(max - min).muli(scale);
    }

    public int getLabel(int labelIndex) {
        if (labelIndex >= size) {
            throw new ArrayIndexOutOfBoundsException("index should be less than data set size: " + size);
        }
        return labels.getInt(labelIndex);
    }

    public BufferedImage getImage(int imageIndex) {
        if (imageIndex >= size) {
            throw new ArrayIndexOutOfBoundsException("index should be less than data set size: " + size);
        }
        DataBuffer buffer;
        buffer = new DataBufferByte(convertToByteArray(data.getRow(imageIndex).dup().data().asInt()), data.columns());

        //3 bytes per pixel: red, green, blue
        WritableRaster raster = Raster.createInterleavedRaster(buffer, IMAGE_WIDTH, IMAGE_HEIGHT, 3 * IMAGE_WIDTH, 3, new int[] {0, 1, 2}, null);
        ColorModel cm = new ComponentColorModel(ColorModel.getRGBdefault().getColorSpace(), false, true, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
        BufferedImage image = new BufferedImage(cm, raster, true, null);
        return image;
    }

    private byte[] convertToByteArray(int[] arr) {
        byte[] res = new byte[3072];
        for (int i = 0; i < res.length ; i++) {
            res[i] = (byte) (arr[i]);
        }
        return res;
    }

}
