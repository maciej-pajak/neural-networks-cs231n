package pl.maciejpajak.playground;

import org.knowm.xchart.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Minimal classifier case study.
 *
 * Based on cs231n.
 */
public class ClassifierCaseStudy {

    public static void main(String[] args) {
        // chart to plot data
        final XYChart chart = new XYChartBuilder()
                .width(600).height(400).title("Area Chart").xAxisTitle("X").yAxisTitle("Y").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);

        // generate data that is not lineary separable
        final int pointsPreClass = 100;
        final int dimensionality = 2;
        final int numberOfClasses = 3;

        INDArray dataSet = Nd4j.zeros(pointsPreClass * numberOfClasses, dimensionality);
        INDArray dataLabels = Nd4j.zeros(pointsPreClass * numberOfClasses, 1);

        for (int i = 0 ; i < numberOfClasses ; i++) {
            // INDArray ix = Nd4j.linspace(pointsPreClass * i, pointsPreClass * (i + 1) - 1, pointsPreClass);
            INDArray radius = Nd4j.linspace(0.0, 1,pointsPreClass);
            INDArray theta = Nd4j
                    .linspace(i * 4, (i + 1) * 4, pointsPreClass)
                    .addi(Nd4j.randn(1, pointsPreClass).muli(0.2));
            INDArray coord = Nd4j.vstack(radius.mul(Transforms.sin(theta)), radius.mul(Transforms.cos(theta))).transpose();
            chart.addSeries("class " + (i + 1),
                    radius.mul(Transforms.sin(theta)).data().asDouble(),
                    radius.mul(Transforms.cos(theta)).data().asDouble());

            for (int j = 0 ; j < pointsPreClass ; j++) { // TODO refactor to utilize Nd4j without loop
                dataSet.putRow(i * pointsPreClass + j, coord.getRow(j));
                dataLabels.putScalar(i * pointsPreClass + j, i);
            }
        }

        // display chart
        new SwingWrapper(chart).displayChart();
    }

}
