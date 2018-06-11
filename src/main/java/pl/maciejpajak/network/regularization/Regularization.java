package pl.maciejpajak.network.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public enum Regularization {

    L1 {
        @Override
        public double calcRegLoss(INDArray weights) {
            return weights.sumNumber().doubleValue();
        }
    },
    L2 {
        @Override
        public double calcRegLoss(INDArray weights) {
            return Transforms.pow(weights, 2).sumNumber().doubleValue();
        }
    };

    public abstract double calcRegLoss(INDArray weights);
}
