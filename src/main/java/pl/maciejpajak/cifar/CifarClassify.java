package pl.maciejpajak.cifar;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.maciejpajak.classifier.LinearClassifier;

import java.io.*;

public class CifarClassify {

    private static final int IMAGES_IN_FILE = 10000;
    private static final int IMAGE_LEN = 3072;

    private CifarClassify() {
    }

    public static Pair<INDArray, INDArray> getCifarLabelsAndData(File... files) {
        INDArray dataSet = Nd4j.create(IMAGES_IN_FILE * files.length, IMAGE_LEN);
        INDArray lables = Nd4j.create(IMAGES_IN_FILE * files.length, 1);
        int imgCount = 0;
        try {
            FileInputStream fis;
            byte[] buffer = new byte[3072];
            for (File f : files) {
                fis = new FileInputStream(f);
                for (int j = 0; j < IMAGES_IN_FILE; j++) {
                    lables.putScalar(imgCount, fis.read());

                    fis.read(buffer);

                    for (int i = 0; i < IMAGE_LEN; i++) {
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

        Pair<INDArray, INDArray> testData = CifarClassify.getCifarLabelsAndData(testDataFile);

        findBestParams(trainData, testData);
    }

    private static void findBestParams(Pair<INDArray, INDArray> trainData, Pair<INDArray, INDArray> testData) {
        double[] learningRates = {0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001};
        double[] regularization = {0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 2500, 5000, 10000, 100000};
        int[] batchSize = {1, 4, 16, 32, 64, 128, 256, 512};

        try (PrintWriter pw = new PrintWriter(new FileWriter("results.txt", true), true)) {

            pw.println("learning_rate, regularization, batchSize, train_acc, test_acc");

            for (int i = 0 ; i < learningRates.length ; i++) {
                for (int j = 0 ; j < regularization.length ; j++) {
                    for (int k = 0 ; k < batchSize.length ; k++) {

                        LinearClassifier lc = LinearClassifier.trainNewLinearClassifier(trainData.getValue(), trainData.getKey(),
                                learningRates[i], regularization[j], 5000, batchSize[k], LinearClassifier.LossFunction.SVM);

                        // 0.0000001, 2500, 2000, 64


                        INDArray predTrain = lc.predict(trainData.getValue());
                        INDArray predTest = lc.predict(testData.getValue());

                        double trainAcc = predTrain.eq(trainData.getKey()).sumNumber().doubleValue() / predTrain.length();
                        double testAcc = predTest.eq(testData.getKey()).sumNumber().doubleValue() / predTest.length();

                        pw.println(String.format("%f, %f, %d, %f, %f", learningRates[i], regularization[j], batchSize[k], trainAcc, testAcc));

                        lc.saveLearningAnalysis(String.format("%f-%f-%d", learningRates[i], regularization[j], batchSize[k]));
                    }
                }
            }


        } catch (IOException e) {
            e.printStackTrace();
        }


    }

}