package pl.maciejpajak.cifar;

import org.nd4j.linalg.api.ndarray.INDArray;
import pl.maciejpajak.cifar.util.CifarDataSet;
import pl.maciejpajak.classifier.LinearClassifier;

import java.io.File;
import java.util.Arrays;
import java.util.logging.Logger;

public class CifarClassify {

    private final static Logger LOG = Logger.getLogger(CifarClassify.class.getName());

    private static final int IMAGES_IN_FILE = 10000;
    private static final int IMAGE_LEN = 3072;

    private CifarClassify() {}

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

        CifarDataSet trainSet = CifarDataSet.loadFromDisk(IMAGES_IN_FILE, IMAGE_LEN, dataSetsFiles);
        CifarDataSet testSet = CifarDataSet.loadFromDisk(IMAGES_IN_FILE, IMAGE_LEN, testDataFile);

        // Preprocessing: subtract the mean image
        INDArray meanImage = trainSet.getMeanExample();
        trainSet.preprocessWithMean(meanImage);
        testSet.preprocessWithMean(meanImage);

        // Split the data into train, val, and test sets. In addition create
        // a small development set as a subset of the training data;
        // this can be used for development so the code runs faster.
        int numTraining = 49000;
        int numValidation = 1000;
        int numTest = 1000;
        int numDev = 500;

        // Validation set will be numValidation points from the original training set.
        CifarDataSet validationSet = trainSet.getSubSet(numTraining, numTraining + numValidation);

        // Our training set will be the first num_train points from the original training set.
        CifarDataSet trainingSet = trainSet.getSubSet(1 , numTraining + 1);

        // Development set - a small subset of the training set.
        CifarDataSet devSet = trainingSet.getSubSet(1, numDev + 1); // TODO change to random; mask = np.random.choice(num_training, num_dev, replace=False)

        // The first numTest points of the original test set as the testing set.
        CifarDataSet testingSet = testSet.getSubSet(1, numTest + 1);

        // Check dimensions
        LOG.info("validation set data : " + Arrays.toString(validationSet.getData().shape()));
        LOG.info("validation set labels : " + Arrays.toString(validationSet.getLabels().shape()));

        LOG.info("training set data : " + Arrays.toString(trainingSet.getData().shape()));
        LOG.info("training set labels : " + Arrays.toString(trainingSet.getLabels().shape()));

        LOG.info("dev set data : " + Arrays.toString(devSet.getData().shape()));
        LOG.info("dev set labels : " + Arrays.toString(devSet.getLabels().shape()));

        LOG.info("testing set data : " + Arrays.toString(testingSet.getData().shape()));
        LOG.info("testing set labels : " + Arrays.toString(testingSet.getLabels().shape()));

        findBestParams(trainingSet, validationSet);


        LinearClassifier lc = LinearClassifier.trainNewLinearClassifier(trainSet.getData(), trainSet.getLabels(),
                0.0000001, 40000, 10000, 256, LinearClassifier.LossFunction.SVM);

//        lc.plotLearningAnalysis();

        INDArray predTrain = lc.predict(trainingSet.getData());
        INDArray predVal = lc.predict(validationSet.getData());

        double trainAcc = predTrain.eq(trainingSet.getLabels()).sumNumber().doubleValue() / predTrain.length();
        double valAcc = predVal.eq(validationSet.getLabels()).sumNumber().doubleValue() / predVal.length();

        System.out.println("Training accuracy: " + trainAcc);
        System.out.println("Validation accuracy: " + valAcc);


    }

    /**
     * Function search for best parameters. Chooses the best hyperparameters by tuning on the validation
     * set. For each combination of hyperparameters, trains a linear SVM on the
     * training set, computes its accuracy on the training and validation sets.
     * In addition, store the best validation accuracy and corredpoding parameters.
     *
     * @param trainingSet
     * @param validationSet
     */
    private static void findBestParams(CifarDataSet trainingSet, CifarDataSet validationSet) {
        double[] learningRates = {0.0000001, 0.00001};
        double[] regularization = {2500, 50000, 25000, 50000};
        int[] batchSize = {128, 256};
        int numIterations = 5000;

        double bestParams[] = new double[3];
        double bestValidationAccuracy = -1.0;

        for (int i = 0 ; i < learningRates.length ; i++) {
            for (int j = 0 ; j < regularization.length ; j++) {
                for (int k = 0 ; k < batchSize.length ; k++) {

                    LinearClassifier lc = LinearClassifier.trainNewLinearClassifier(trainingSet.getData(), trainingSet.getLabels(),
                            learningRates[i], regularization[j], numIterations, batchSize[k], LinearClassifier.LossFunction.SVM);

                    INDArray predTrain = lc.predict(trainingSet.getData());
                    INDArray predVal = lc.predict(validationSet.getData());

                    double trainAcc = predTrain.eq(trainingSet.getLabels()).sumNumber().doubleValue() / predTrain.length();
                    double valAcc = predVal.eq(validationSet.getLabels()).sumNumber().doubleValue() / predVal.length();

                    if (valAcc > bestValidationAccuracy) {
                        bestValidationAccuracy = valAcc;
                        bestParams = new double[] {learningRates[i], regularization[j], batchSize[k]};
                    }

                    LOG.info(String.format("rate = %.7f, reg = %f, batch = %d, train_acc = %f, val_acc = %f", learningRates[i], regularization[j], batchSize[k], trainAcc, valAcc));
                }
            }
        }

        LOG.info(String.format("Best validation accuracy %f for learning_rate = %.7f, reg = %f, batch_size = %.0f", bestValidationAccuracy, bestParams[0], bestParams[1], bestParams[2]));

    }

}