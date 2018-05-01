package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.util.Nd4jHelper;

import java.util.Arrays;

public class MulticlassSVMLoss implements ILossFunction {

    private static final Logger logger = LoggerFactory.getLogger(MulticlassSVMLoss.class);

    private INDArray scoresTmp;

    @Override
    public double calculateScore(INDArray data, INDArray labels, boolean average) {
        int samples = data.size(0);

        INDArray correctClassesScore = Nd4jHelper.getSpecifiedElements(data, labels);
        INDArray score = Transforms.max(data.subColumnVector(correctClassesScore).add(1.0),0);

        // set correct class score to 0
        Nd4jHelper.putScalar(score, labels, 0);
        this.scoresTmp = score;
        double loss = score.sumNumber().doubleValue();

        if (average) {
            loss /= samples;
        }
        return loss;
    }

    @Override
    public INDArray calculateGradient(INDArray lables) {
        INDArray binary = scoresTmp.gt(0);
        INDArray rowSums = binary.sum(1);

        // set correct class coefficint to -rowSums
        Nd4jHelper.putValues(binary, lables, rowSums.mul(-1));

        INDArray dScores = binary;
//        INDArray dScores = data.transpose().mmul(binary);

        // average
        dScores.divi(scoresTmp.size(0));
        return dScores;
    }

}
