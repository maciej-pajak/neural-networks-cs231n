package pl.maciejpajak.playground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class LinerClassifierPlayground {
    
    public static double f(INDArray x, int correctClass, INDArray W) {
        INDArray dot = W.mmul(x.transpose());

        double delta = 1.0;
        INDArray margins = Transforms.max(dot.sub(dot.getDouble(correctClass)).add(delta), 0);
        margins.putScalar(correctClass, 0.0);

        return margins.sumNumber().doubleValue();
    }

    /**
     * SVM loss function without regularization. Fully vectorized implementation.
     *
     * Based on Stanford cs231n course.
     *
     * @param W - weights (e.g. 10 x 3073)
     * @param y - array of integers specifying correct class (e.g. 50,000-D array)
     * @param x - holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
     * @return - loss
     */
    public static double L(INDArray W, INDArray y, INDArray x) {
        // W * x
        INDArray dot = W.mmul(x);

        // indexes of scores of correct classes in dot array
        INDArray indexesArray = Nd4j.linspace(0, x.size(1) - 1, x.size(1))
                                        .mul(dot.size(0))
                                        .add(y);

        // vector with corect class score for each training example
        INDArray correctClassScore = Nd4j.toFlattened(dot.transpose()).get(new SpecifiedIndex(indexesArray.data().asInt()));
        System.out.println(correctClassScore);

        double delta = 1.0;

        INDArray margins = Transforms.max(dot.transpose().subColumnVector(correctClassScore).add(delta),0);

        double loss = (margins.sumNumber().doubleValue() - y.size(1) * delta);

        return loss;

    }

    public static void main(String[] args) {
        System.out.println(Nd4j.linspace(0, 9, 20));
        // output: [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
        System.out.println();



//        - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
//        - y is array of integers specifying correct class (e.g. 50,000-D array)
//        - W are weights (e.g. 10 x 3073)
        INDArray x = Nd4j.ones(4, 10); // 10 - training examples
        INDArray y = Nd4j.ones(1, 10);
        INDArray W = Nd4j.ones(5, 4).mul(3); // 5 - classes

        System.out.println(x);
        System.out.println(W);
        System.out.println(L(W, y, x));

    }

}
