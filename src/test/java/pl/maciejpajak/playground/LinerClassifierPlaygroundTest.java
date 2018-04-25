package pl.maciejpajak.playground;

import javafx.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.maciejpajak.classifier.LinearClassifier;
import pl.maciejpajak.classifier.LossFunction;

import static org.junit.Assert.*;

public class LinerClassifierPlaygroundTest {

    @Test
    public void l() {
        INDArray x = Nd4j.create(new double[]{0.1, 0.7, 1.3, 1.9, 2.5,
                0.2, 0.8 ,1.4, 2.0, 2.6,
                0.3, 0.9, 1.5, 2.1, 2.7,
                0.4, 1.0, 1.6, 2.2, 2.8}, new int[]{4, 5});
        INDArray y = Nd4j.create(new double[]{0, 1, 2, 3, 4}, new int[]{5, 1});
        INDArray W = Nd4j.create(new double[]{0.1, 0.6, 1.1, 1.6,
                0.2, 0.7, 1.2, 1.7,
                0.3, 0.8, 1.3, 1.8,
                0.4, 0.9, 1.4, 1.9,
                0.5, 1.0, 1.5, 2.0}, new int[]{5, 4});
        LossFunction.SVM.loss(x.transpose(),y,W.transpose(), 0);
//        assertEquals(16.86, LinerClassifierPlayground.L(W, y, x.transpose()), 0.01);
        Pair<Double, INDArray> resNaive = LossFunction.SVM_NAIVE.loss(x.transpose(),y,W.transpose(),10);

        Pair<Double, INDArray> resVec = LossFunction.SVM.loss(x.transpose(),y,W.transpose(),10);
        assertEquals(16.86 / 5 + 280, resNaive.getKey(), 0.01);
        assertEquals(16.86 / 5 + 280, resVec.getKey(), 0.01);
        assertEquals(resNaive.getValue(), resVec.getValue());
        INDArray gradNumSvm = LossFunction
                .numericalGradient(LossFunction.SVM, x.transpose(),y,W.transpose(),0, 0.0001);
        INDArray gradNumSvmNaive = LossFunction
                .numericalGradient(LossFunction.SVM_NAIVE, x.transpose(),y,W.transpose(),0, 0.0001);
        assertArrayEquals(gradNumSvm.sub(resVec.getValue()).data().asDouble(), Nd4j.zeros(gradNumSvm.shape()).data().asDouble(), 0.01);
        assertArrayEquals(gradNumSvmNaive.data().asDouble(), resNaive.getValue().data().asDouble(), 0.01);
    }
}