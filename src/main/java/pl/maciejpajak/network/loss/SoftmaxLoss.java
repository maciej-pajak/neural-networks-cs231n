package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import pl.maciejpajak.util.Nd4jHelper;

public class SoftmaxLoss implements ILossFunction {

    private INDArray probsTmp;

    @Override
    public double calculateScore(INDArray data, INDArray labels, boolean average) {
        INDArray correctLogProbs = calculateScoresArray(data, labels);
        double loss = correctLogProbs.sumNumber().doubleValue();
        if (average) loss /= data.size(0);
        return loss;
    }

    private INDArray calculateScoresArray(INDArray data, INDArray labels) {
        // compute the class probabilities
        INDArray expScores = Transforms.exp(data);
        INDArray probs = expScores.divColumnVector(expScores.sum(1));
        this.probsTmp = probs;

        // compute the loss: average cross-entropy loss and regularization
        INDArray correctLogProbs = Transforms.log(Nd4jHelper.getSpecifiedElements(probs, labels)).mul(-1);
        return correctLogProbs;
    }

    @Override
    public INDArray calculateGradient(INDArray labels) {
        INDArray dScores = probsTmp;
        Nd4jHelper.putValues(dScores, labels, Nd4jHelper.getSpecifiedElements(dScores, labels).neg().add(1.0));
        return dScores;
    }

}
