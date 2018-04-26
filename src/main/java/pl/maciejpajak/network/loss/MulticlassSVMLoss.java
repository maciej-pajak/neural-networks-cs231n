package pl.maciejpajak.network.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import pl.maciejpajak.util.Nd4jHelper;

public class MulticlassSVMLoss implements ILossFunction {

    @Override
    public double calculateScore(INDArray data, INDArray labels, boolean average) {
        int samples = data.size(0);
        INDArray correctClassesScore = Nd4jHelper.getSpecifiedElements(data, labels);
        INDArray margins = Transforms.max(data.subColumnVector(correctClassesScore).add(1.0),0);

        double loss = margins.sumNumber().doubleValue() - samples;
        if (average) {
            loss /= samples;
        }
        return loss;
    }

}
