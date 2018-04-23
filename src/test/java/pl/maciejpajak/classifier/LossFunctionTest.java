package pl.maciejpajak.classifier;

import javafx.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

public class LossFunctionTest {

    @Test
    public void crossTestSvmLossFunctions() {

        INDArray input = Nd4j.randn(5, 20);
        INDArray labels = Nd4j.ones(5,1 );
        INDArray weights = Nd4j.randn(20,10);
        Pair<Double, INDArray> svmResult = LossFunction.SVM.loss(input, labels, weights, 1000);
        Pair<Double, INDArray> svmNaiveResult = LossFunction.SVM_NAIVE.loss(input, labels, weights, 1000);

        assertTrue(svmResult.getValue().equalsWithEps(svmNaiveResult.getValue(), 0.001));
    }

    @Test
    public void svmCheckAnalyticalGradientTest() {
        INDArray input = Nd4j.randn(5, 20);
        INDArray labels = Nd4j.ones(5,1 );
        INDArray weights = Nd4j.randn(20,10);
        Pair<Double, INDArray> svmResult = LossFunction.SVM.loss(input, labels, weights, 1000);
        Pair<Double, INDArray> svmNaiveResult = LossFunction.SVM_NAIVE.loss(input, labels, weights, 1000);

        INDArray svmNumericalGradient = LossFunction.numericalGradient(LossFunction.SVM, input, labels, weights, 1000, 0.001);
        INDArray svmNaiveNumericalGradient = LossFunction.numericalGradient(LossFunction.SVM_NAIVE, input, labels, weights, 1000, 0.001);

        System.out.println(svmNumericalGradient.getRow(0));
        System.out.println(svmResult.getValue().getRow(0));

        assertTrue(svmNumericalGradient.equalsWithEps(svmResult.getValue(), 10));
        assertTrue(svmNaiveNumericalGradient.equalsWithEps(svmNaiveResult.getValue(), 10));
    }
}