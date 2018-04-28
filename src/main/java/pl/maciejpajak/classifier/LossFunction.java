package pl.maciejpajak.classifier;

import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import pl.maciejpajak.util.Nd4jHelper;

public enum LossFunction {
    SVM {
        @Override
        public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {

            double loss;
            INDArray dW;
            int samples = batchSet.size(0);

            // X * W
            INDArray scores = batchSet.mmul(weights);
            INDArray correctClassesScore = Nd4jHelper.getSpecifiedElements(scores, batchLabels);
            INDArray margins = Transforms.max(scores.subColumnVector(correctClassesScore).add(1.0),0);

            loss = margins.sumNumber().doubleValue() - samples;
            loss /= samples;
            loss += reg * Transforms.pow(weights, 2).sumNumber().doubleValue();

            // gradient ==============================================
            INDArray binary = margins.gt(0);

            // set correct class binary to 0
            Nd4jHelper.putScalar(binary, batchLabels, 0);
            INDArray rowSums = binary.sum(1);

            // set correct class coefficint to -rowSums
            Nd4jHelper.putValues(binary, batchLabels, rowSums.mul(-1));

            dW = batchSet.transpose().mmul(binary);

            // average
            dW.divi(samples);

            // regularization
            dW.addi(weights.mul(2 * reg));

            return new Pair<>(loss, dW);
        }
    },
    SVM_NAIVE {
        @Override
        public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
            int numClasses = weights.size(1);
            int samples = batchSet.size(0);

            INDArray dW = Nd4j.zeros(weights.shape());

            double loss = 0;
            INDArray scores;
            double correctClassScore;
            double margin;

            // compute loss and gradient
            for (int i = 0 ; i < samples ; i++) {
                scores = batchSet.getRow(i).mmul(weights);
                correctClassScore = scores.getDouble(batchLabels.getInt(i));

                for (int j = 0 ; j < numClasses ; j++) {
                    if (j == batchLabels.getInt(i)) continue;
                    margin = scores.getDouble(j) - correctClassScore + 1;
                    if (margin > 0) {
                        loss += margin;
                        dW.getColumn(batchLabels.getInt(i)).subi(batchSet.getRow(i).transpose());
                        dW.getColumn(j).addi(batchSet.getRow(i).transpose());
                    }
                }
            }
            // average over all examples
            loss /= samples;
            dW.divi(samples);

            // add regularization
            loss += reg * Transforms.pow(weights, 2).sumNumber().doubleValue();
            dW.addi(weights.mul(2 * reg));

            return new Pair<>(loss, dW);
        }
    },
    SOFTMAX {
        @Override
        public Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg) {
            double loss;
            INDArray dW;
            int samples = batchSet.size(0);

            // X * W
            INDArray scores = batchSet.mmul(weights);

            // unnormalized probabilities
            INDArray expScores = Transforms.exp(scores);

            // normalize
            INDArray probs = expScores.divColumnVector(expScores.sum(1));

            INDArray correctLogProbs = Transforms.log(Nd4jHelper.getSpecifiedElements(probs, batchLabels)).mul(-1);

            // compute the loss - average cross-entropy loss and regularization

            loss = correctLogProbs.sumNumber().doubleValue() / samples;
            loss += reg * Transforms.pow(weights, 2).sumNumber().doubleValue();

            // gradient ==============================================

            INDArray dScores = probs; // or dup?

            // update correct class probabilities
            Nd4jHelper.putValues(dScores, batchLabels, Nd4jHelper.getSpecifiedElements(dScores, batchLabels).neg().add(1.0));

            // average
            dScores.divi(samples);

            dW = batchSet.transpose().mmul(dScores);

            // regularization
            dW.addi(weights.mul(2 * reg));

            return new Pair<>(loss, dW);
        }
    };

    /**
     * Compute the loss function and its derivative.
     *
     * @param batchSet - array of shape N x D containing a minibatch of N data points, each point has dimension D.
     * @param batchLabels - row vector of length N containing labels for the minibatch.
     * @param weights - array of shape D x C containing weights.
     * @param reg - regularization strength
     *
     * @return - a pair where the key is loss and value is loss gradient with respect to weights W (shape N x D)
     */
    public abstract Pair<Double, INDArray> loss(INDArray batchSet, INDArray batchLabels, INDArray weights, double reg);

    /**
     * Calculates gradient of loss function with respect to weights using numerical method.
     *
     * @param lossFunction - loss function
     * @param batchSet - lossFunction argument - array of shape N x D containing a minibatch of N data points, each point has dimension D.
     * @param batchLabels - lossFunction argument - row vector of length N containing labels for the minibatch.
     * @param weights - lossFunction argument - array of shape D x C containing weights.
     * @param reg - lossFunction argument - regularization strength
     * @param h - delta weight
     *
     * @return - gradient array
     */
    public static INDArray numericalGradient(LossFunction lossFunction, INDArray batchSet, INDArray batchLabels, INDArray weights, double reg, double h) {
        INDArray dW = Nd4j.zeros(weights.shape());

        double lossPos;
        double lossNeg;
        double grad;

        double tmp;

        for (int i = 0 ; i < weights.rows() ; i++) {
            for (int j = 0 ; j < weights.columns() ; j++) {
                tmp = weights.getDouble(i, j);
                weights.putScalar(i, j, tmp - h);
                lossNeg = lossFunction.loss(batchSet, batchLabels, weights, reg).getKey();
                weights.putScalar(i, j, tmp + h);
                lossPos = lossFunction.loss(batchSet, batchLabels, weights, reg).getKey();
                weights.putScalar(i, j, tmp);

                grad = (lossPos - lossNeg) / (2 * h);
                dW.putScalar(i, j, grad);
            }
        }

        return dW;
    }
}

