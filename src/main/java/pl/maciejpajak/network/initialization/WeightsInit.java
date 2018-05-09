package pl.maciejpajak.network.initialization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public enum WeightsInit {

    XAVIER {
        @Override
        public INDArray initialize(int[] shape) {
            return Nd4j.randn(shape).divi(Math.sqrt(shape[0]));
        }
    },
    XAVIER_RELU {
        @Override
        public INDArray initialize(int[] shape) {
            return Nd4j.randn(shape).muli(Math.sqrt(2.0 / shape[0]));
        }
    },
    SMALL_RANDOM {
        @Override
        public INDArray initialize(int[] shape) {
            return Nd4j.randn(shape).muli(0.0001);
        }
    };

    public abstract INDArray initialize(int[] shape);

}
