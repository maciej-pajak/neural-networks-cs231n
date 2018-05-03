package pl.maciejpajak.cifar;

import pl.maciejpajak.classifier.LearningHistory;
import pl.maciejpajak.network.SimpleNetwork;
import pl.maciejpajak.network.activation.Identity;
import pl.maciejpajak.network.activation.ReLU;
import pl.maciejpajak.network.loss.MulticlassSVMLoss;

public class CifarTwoLayerNetwork {

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
                .loss(new MulticlassSVMLoss())
                .learningRate(1e-7)
                .regularization(1e-1)
                .iterations(500)
                .batchSize(256).build();

        LearningHistory history = network.train(trainingSet, validationSet);

        history.plot();

    }

}
