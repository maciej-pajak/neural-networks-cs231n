package pl.maciejpajak.cifar;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.SimpleNetwork;
import pl.maciejpajak.network.activation.Identity;
import pl.maciejpajak.network.activation.ReLU;
import pl.maciejpajak.network.loss.SoftmaxLoss;

public class CifarTwoLayerNetwork {

    public static final Logger logger = LoggerFactory.getLogger(CifarTwoLayerNetwork.class);

    public static void main(String[] args) {
        CifarDataLoader loader = new CifarDataLoader();
        loader.load();

        CifarDataSet validationSet = loader.getValidationSet();

        // Our training set will be the first num_train points from the original training set.
        CifarDataSet trainingSet = loader.getTrainingSet();

        // Development set - a small subset of the training set.
        CifarDataSet devSet = loader.getDevSet();

        // The first numTest points of the original test set as the testing set.
        CifarDataSet testingSet = loader.getTestingSet();

        SimpleNetwork network = SimpleNetwork.builder()
                .layer(3072, 100, new ReLU())
                .layer(100, 10, new Identity())
                .loss(new SoftmaxLoss())
                .learningRate(1e-4)
                .learningRateDecay(0.95)
//                .regularization(0.1)
                .regularization(0.01)
                .iterations(1000)
                .batchSize(256).build();

        LearningHistory history = network.train(trainingSet, validationSet);

        logger.info("Training accuracy:   {}", network.checkAccuracy(trainingSet));
        logger.info("Validation accuracy: {}", network.checkAccuracy(validationSet));
        logger.info("Testing accuracy:    {}", network.checkAccuracy(testingSet));
        history.plot();
    }

}
