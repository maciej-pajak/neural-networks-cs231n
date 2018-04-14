package pl.maciejpajak.classifier;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.logging.Logger;

/**
 * Linear classifier implementation.
 * Based on Stanford cs231n course.
 */
public class LinearClassifier {

    private final static Logger LOG = Logger.getLogger(LinearClassifier.class.getName());

    private INDArray weights;
    private LossFunction lossFunction;

    private LinearClassifier(INDArray weights, LossFunction lossFunction) {
        this.weights = weights;
        this.lossFunction = lossFunction;
    }

    /**
     * Trains new linear classifier using stochastic gradient descent.
     *
     *
     * @param trainingSet - array of shape N x D containing training data (N training samples each of dimension D)
     * @param trainingLabels - vector of length N containing training labels
     * @param learningRate - learning rate for optimization
     * @param reg - regularization strength
     * @param iterations - number iterations when optimizing
     * @param batchSize - number of training examples to use at each iteration
     * @param lossFunction - loss function to be used in calculations
     *
     * @return - trained linear classifier
     */
    public static LinearClassifier trainNewLinearClassifier(INDArray trainingSet, INDArray trainingLabels,
                                                            double learningRate, double reg, int iterations, int batchSize,
                                                            LossFunction lossFunction) {

        INDArray weights = Nd4j.create(0,0);
        //    num_train, dim = X.shape
        //            num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        //    if self.W is None:
        //            # lazily initialize W
        //    self.W = 0.001 * np.random.randn(dim, num_classes)
        //

        // Run stochastic gradient descent to optimize W

        INDArray batchSet;
        INDArray batchLabels;

        for (int i = 0 ; i < iterations ; i++) {

            // batchSet =
            // batchLabels =

            //            # Sample batch_size elements from the training data and their           #
            //            # corresponding labels to use in this round of gradient descent.        #
            //            # Store the data in X_batch and their corresponding labels in           #
            //            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            //            # and y_batch should have shape (batch_size,)                           #
            //            #                                                                       #
            //            # Hint: Use np.random.choice to generate indices. Sampling with         #
            //            # replacement is faster than sampling without replacement.              #

            // evaluate loss and gradient
            // Pair<Double, INDArray> lossAndGradient = lossFunction.loss(...);

            // double loss;
            // perform parameter update

            // Update the weights using the gradient and the learning rate.
            if (iterations % 100 == 0) {
               // LOG.log(Level.INFO, String.format("iteration %d / %d: loss %f", i, iterations, loss));
            }
        }

        return new LinearClassifier(weights, lossFunction);
    }

    /**
     * Predicts lables for input data based on learned weights.
     *
     * @param dataSet - array of shape N x D containing training data (N training samples each of dimension D)
     * @return - Predicted labels for the data in X. y_pred is a 1-dimensional array of length N,
     *          and each element is an integer giving the predicted class.
     */
    public INDArray predict(INDArray dataSet) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public enum LossFunction {
        SVM {
            @Override
            Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
                return null;
            }
        },
        SOFTMAX {
            @Override
            Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
                throw new UnsupportedOperationException("Not yet implemented");
            }
        };

        /**
         * Compute the loss function and its derivative.
         *
         * @param batchSet - array of shape N x D containing a minibatch of N data points, each point has dimension D.
         * @param batchLabels - array of shape (N,) containing labels for the minibatch.
         * @param weights - array of shape D x C containing weights.
         * @param reg - regularization strength
         *
         * @return - a pair where the key is loss and value is loss gradient with respect to weights W (shape N x D)
         */
        abstract Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg);
    }

}
