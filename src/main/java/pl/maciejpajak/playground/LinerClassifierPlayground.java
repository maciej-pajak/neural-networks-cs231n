package pl.maciejpajak.playground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class LinerClassifierPlayground {

    public static double f(INDArray x, int correctClass, INDArray W) {
        System.out.println();

        INDArray dot = W.mmul(x.transpose());
        System.out.println(dot);

        double delta = 1.0;
        INDArray margins = Transforms.max(dot.sub(dot.getDouble(correctClass)).add(delta), 0);
        margins.putScalar(correctClass, 0.0);
        System.out.println(margins);

        return margins.sumNumber().doubleValue();
    }

    public static double L(INDArray W, INDArray y, INDArray x) {
        /*
        fully-vectorized implementation :
        - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
        - y is array of integers specifying correct class (e.g. 50,000-D array)
        - W are weights (e.g. 10 x 3073)
        */

        // W * x
        INDArray dot = W.mmul(x);

        System.out.println(dot);
        System.out.println(dot.transpose());
        System.out.println("W " + Arrays.toString(W.shape()));
        System.out.println("x " + Arrays.toString(x.shape()));
        System.out.println("dot " + Arrays.toString(dot.shape()));

        // indexes of scores of correct classes in dot array
        INDArray indexesArray = Nd4j.linspace(0, x.size(1) - 1, x.size(1))
                                        .mul(dot.size(0))
                                        .add(y);

        System.out.println(indexesArray);

        // vector with corect class score for each training example
        INDArray correctClassScore = Nd4j.toFlattened(dot.transpose()).get(new SpecifiedIndex(indexesArray.data().asInt()));

        System.out.println(correctClassScore);
        double delta = 1.0;
        System.out.println("======");
        System.out.println(dot.transpose().subColumnVector(correctClassScore));

        INDArray margins = Transforms.max(dot.transpose().subColumnVector(correctClassScore).add(delta),0);

        System.out.println(margins);
//        INDArray m;
//        INDArray these_indexes = Nd4j.linspace(0, m.size(0)-1, m.size(0)).mul(mn.size(1)).add(m.transpose());
//        INDArray result = Nd4j.toFlattened(mn).get(new SpecifiedIndex(these_indexes.data().asInt())).transpose();
        double loss = (margins.sumNumber().doubleValue() - y.size(1) * delta);
        System.out.println(loss);
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

//        System.out.println(W.sumNumber());
//        System.out.println(x);
//        System.out.println(W);
//        System.out.println("loss = " + f(x, 1, W));
    }

}
