package pl.maciejpajak.cifar;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.maciejpajak.classifier.LinearClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class CifarClassify {

    private static final int IMAGES_IN_FILE = 10000;
    private static final int IMAGE_LEN = 3072;

    private CifarClassify() {}

    public static Pair<INDArray, INDArray> getCifarLabelsAndData(File... files) {
        INDArray dataSet = Nd4j.create(IMAGES_IN_FILE * files.length, IMAGE_LEN);
        INDArray lables = Nd4j.create(IMAGES_IN_FILE * files.length, 1);
        int imgCount = 0;
        try {
            FileInputStream fis;
            byte[] buffer = new byte[3072];
            for (File f : files) {
                fis = new FileInputStream(f);
                for (int j = 0 ; j < IMAGES_IN_FILE ; j++) {
                    lables.putScalar(imgCount, fis.read());

                    fis.read(buffer);

                    for (int i = 0 ; i < IMAGE_LEN ; i++) {
                        dataSet.putScalar(imgCount, i, (double) buffer[i]);
                    }
                    imgCount++;

                }
                fis.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new Pair<>(lables, dataSet);
    }

    private static final String PATH = "/Users/mac/Downloads/cifar-10-batches-bin";

    public static void main(String[] args) {
        File[] dataSetsFiles = {new File(PATH + "/data_batch_1.bin"),
                new File(PATH + "/data_batch_2.bin"),
                new File(PATH + "/data_batch_3.bin"),
                new File(PATH + "/data_batch_4.bin"),
                new File(PATH + "/data_batch_5.bin")
        };

        File testDataFile = new File(PATH + "/test_batch.bin");

//        learning_rates = [1e-7, 5e-5]
//        regularization_strengths = [2.5e4, 5e4]

        Pair<INDArray, INDArray> trainData = CifarClassify.getCifarLabelsAndData(dataSetsFiles);

//        Pair<INDArray, INDArray> testData = CifarClassify.getCifarLabelsAndData(testDataFile);

        LinearClassifier.trainNewLinearClassifier(trainData.getValue(), trainData.getKey(), 0.00001, 2500, 100, 128, LinearClassifier.LossFunction.SVM);

    }
}
