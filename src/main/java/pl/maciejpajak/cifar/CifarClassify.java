package pl.maciejpajak.cifar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.SimpleNetwork;
import pl.maciejpajak.network.activation.Identity;
import pl.maciejpajak.network.loss.MulticlassSVMLoss;
import pl.maciejpajak.util.ImageDisplayer;
import pl.maciejpajak.classifier.LinearClassifier;
import pl.maciejpajak.classifier.LossFunction;

import java.io.File;
import java.util.Random;

public class CifarClassify {

    private final static Logger logger = LoggerFactory.getLogger(CifarClassify.class);

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

        CifarDataSet trainSet = CifarDataSet.loadFromDisk(IMAGES_IN_FILE, IMAGE_LEN, dataSetsFiles);
        CifarDataSet testSet = CifarDataSet.loadFromDisk(IMAGES_IN_FILE, IMAGE_LEN, testDataFile);

        // Preprocessing: subtract the mean image
//        INDArray meanImage = trainSet.getMeanExample();
//        trainSet.preprocessWithMean(meanImage);
//        testSet.preprocessWithMean(meanImage);

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

//        // Display images to validate
//        ImageDisplayer imageDisplayer = new ImageDisplayer("Check images", 2,2);
//        imageDisplayer.addImage(String.valueOf(validationSet.getLabel(0)) + " 1", validationSet.getImage(0));
//        imageDisplayer.addImage(String.valueOf(trainingSet.getLabel(0)) + " 2", trainingSet.getImage(0));
//        imageDisplayer.addImage(String.valueOf(devSet.getLabel(0)) + " 3", devSet.getImage(0));
//        imageDisplayer.addImage(String.valueOf(testingSet.getLabel(0)) + " 4", testingSet.getImage(0));
//        imageDisplayer.show();
//
//        // Check dimensions
//        logger.info("validation set data : " + Arrays.toString(validationSet.getData().shape()));
//        logger.info("validation set labels : " + Arrays.toString(validationSet.getLabels().shape()));
//
//        logger.info("training set data : " + Arrays.toString(trainingSet.getData().shape()));
//        logger.info("training set labels : " + Arrays.toString(trainingSet.getLabels().shape()));
//
//        logger.info("dev set data : " + Arrays.toString(devSet.getData().shape()));
//        logger.info("dev set labels : " + Arrays.toString(devSet.getLabels().shape()));
//
//        logger.info("testing set data : " + Arrays.toString(testingSet.getData().shape()));
//        logger.info("testing set labels : " + Arrays.toString(testingSet.getLabels().shape()));

        //findBestParams(trainingSet, validationSet);

        //bestParamsCoarseSearch(devSet, validationSet);

        SimpleNetwork lc = SimpleNetwork.builder()
                                    .layer(3072, 10, new Identity())
                                    .loss(new MulticlassSVMLoss()).build();

        LearningHistory history = lc.train(trainingSet, validationSet,
                0.000001, 10000,2000, 256);

        history.plot();



//        LinearClassifier lc = LinearClassifier.trainNewLinearClassifier(trainingSet.getData(), trainingSet.getLabels(),
//                validationSet.getData(), validationSet.getLabels(),
//                0.000001, 1000, 2000, 200, LossFunction.SVM);
////                0.0000000025, 150000, 5000, 16, LossFunction.SVM);
////                0.0000001, 50000, 5000, 16, LossFunction.SVM);
//
//        lc.getLearningHistory().plot();
//
//        // display template images created from weights
//        CifarDataSet templates = new CifarDataSet(lc.getWeights().transpose(),
//               null);
//        templates.rescale(255);
//        ImageDisplayer id = new ImageDisplayer("Template images", 2, 5);
//        for (int i = 0 ; i < templates.getSize() ; i++) {
//            id.addImage(String.valueOf(i), templates.getImage(i));
//        }
//        id.show();

        // predict
        INDArray predTrain = lc.predict(trainingSet.getData());
        INDArray predVal = lc.predict(validationSet.getData());
        INDArray predTest = lc.predict(testingSet.getData());

        double trainAcc = predTrain.eq(trainingSet.getLabels()).meanNumber().doubleValue();
        double valAcc = predVal.eq(validationSet.getLabels()).meanNumber().doubleValue();
        double testAcc = predTest.eq(testingSet.getLabels()).meanNumber().doubleValue();
        logger.info("Training accuracy: " + trainAcc);
        logger.info("Validation accuracy: " + valAcc);
        logger.info("Testing accuracy: " + testAcc);
    }

    private static void bestParamsCoarseSearch(CifarDataSet dataSet, CifarDataSet valSet) {
        int loops = 100;
        Random r = new Random();
        for (int i = 0 ; i < loops ; i++) {
            double lr = Math.pow(10, r.nextDouble() * (-3) - 3);
            double reg = Math.pow(10, r.nextDouble() * 10 - 5);

            LinearClassifier lc = LinearClassifier.trainNewLinearClassifier(dataSet.getData(), dataSet.getLabels(),
                    lr, reg, 100, 128, LossFunction.SVM);

            INDArray predVal = lc.predict(valSet.getData());
            double valAcc = predVal.eq(valSet.getLabels()).meanNumber().doubleValue();

            logger.info("val_acc: {}, lr: {}, reg: {},  ({}/{})", valAcc, lr, reg, (i + 1), loops);
        }
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
                            learningRates[i], regularization[j], numIterations, batchSize[k], LossFunction.SVM);

                    INDArray predTrain = lc.predict(trainingSet.getData());
                    INDArray predVal = lc.predict(validationSet.getData());

                    double trainAcc = predTrain.eq(trainingSet.getLabels()).sumNumber().doubleValue() / predTrain.length();
                    double valAcc = predVal.eq(validationSet.getLabels()).sumNumber().doubleValue() / predVal.length();

                    if (valAcc > bestValidationAccuracy) {
                        bestValidationAccuracy = valAcc;
                        bestParams = new double[] {learningRates[i], regularization[j], batchSize[k]};
                    }

                    logger.info(String.format("rate = %.7f, reg = %f, batch = %d, train_acc = %f, val_acc = %f", learningRates[i], regularization[j], batchSize[k], trainAcc, valAcc));
                }
            }
        }

        logger.info(String.format("Best validation accuracy %f for learning_rate = %.7f, reg = %f, batch_size = %.0f", bestValidationAccuracy, bestParams[0], bestParams[1], bestParams[2]));

    }

}