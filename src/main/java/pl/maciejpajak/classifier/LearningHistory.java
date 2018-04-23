package pl.maciejpajak.classifier;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * Class stores learningh history and provides methods for plotting and saving results.
 */
public class LearningHistory {

    // column 0 - iteration
    // column 1 - loss
    // column 2 - accuracy
    private final INDArray history;
    private int currentRecord;

    public LearningHistory(int length) {
        this.history = Nd4j.create(length, 3);
        currentRecord = 0;
    }

    public void addNextRecord(int iteration, double loss, double accuracy) {
        if (currentRecord + 1 > history.rows()) throw new IndexOutOfBoundsException("history is full");
        history.putRow(currentRecord++, Nd4j.create(new double[]{iteration, loss, accuracy}, new int[]{1,3}));
    }

    public INDArray getHistory() {
        return history.dup();
    }

    public void plot() {
        // Create Chart
        XYChart chart = new XYChartBuilder().width(600).height(500)
                .title("Learning analysis").xAxisTitle("iteration").yAxisTitle("val").build();

        double[] xIterations = history.getColumn(0).dup().data().asDouble();
        double[] yLoss = history.getColumn(1).dup().data().asDouble();
        // convert loss to percentage to fit on one chart with accuracy
        double[] yAccuracy = history.getColumn(2).div(history.getColumn(2).maxNumber()).dup().data().asDouble();


        // Series
        chart.addSeries("Accuracy [%]", xIterations, yAccuracy);
        chart.addSeries("Loss [%  of max loss]", xIterations, yLoss);

        new SwingWrapper(chart).displayChart();
    }

    public void saveLearningHistoryPlot(String fileName) {
        // Create Chart
        XYChart chart = new XYChartBuilder().width(600).height(500).title("Learning analysis - " + fileName).xAxisTitle("iteration").yAxisTitle("val").build();

        double[] xIterations = history.getColumn(0).dup().data().asDouble();
        double[] yAccuracy = history.getColumn(1).dup().data().asDouble();
        double[] yLoss = history.getColumn(2).div(history.getColumn(2).maxNumber()).dup().data().asDouble();


        // Series
        chart.addSeries("Accuracy [%]", xIterations, yAccuracy);
        chart.addSeries("Loss [%  of max loss]", xIterations, yLoss);

//        new SwingWrapper(chart).displayChart();

        // try to save
        try {
            BitmapEncoder.saveBitmap(chart, fileName, BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
