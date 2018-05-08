package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.util.Nd4jHelper;

public class MulticlassSVMLoss implements ILossFunction {

    private static final Logger logger = LoggerFactory.getLogger(MulticlassSVMLoss.class);

    private INDArray scoresTmp;

    @Override
    public double calculateScore(INDArray data, INDArray labels, boolean average) {
        int samples = data.size(0);

        INDArray correctClassesScore = Nd4jHelper.getSpecifiedElements(data, labels);
        INDArray margins = Transforms.max(data.subColumnVector(correctClassesScore).add(1.0),0);

        // set correct class margin to 0
        Nd4jHelper.putScalar(margins, labels, 0);

        this.scoresTmp = margins;
        double loss = margins.sumNumber().doubleValue();

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
