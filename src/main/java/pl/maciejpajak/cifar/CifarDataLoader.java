package pl.maciejpajak.cifar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.maciejpajak.util.DataSet;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

public class CifarDataLoader {

    private static final Logger logger = LoggerFactory.getLogger(CifarDataLoader.class);

    private static final int IMAGES_IN_FILE = 10000;
    private static final int IMAGE_LEN = 3072;

    // default sizes of subsets
    private int numTraining = 49000;
    private int numValidation = 1000;
    private int numTest = 10000;
    private int numDev = 500;

    private List<String> dataFiles;
    private List<String> testFiles;

    private CifarDataSet trainingSet;
    private CifarDataSet validationSet;
    private CifarDataSet devSet;
    private CifarDataSet testingSet;

    public CifarDataLoader() {
        this.dataFiles = new ArrayList<>();
        this.testFiles = new ArrayList<>();
    }

    public void load() {
        loadProperties();
        DataSet trainSet = loadFromFiles(dataFiles);
        DataSet testSet = loadFromFiles(testFiles);

        // Validation set will be numValidation points from the original training set.
        validationSet = trainSet.getSubSet(numTraining, numTraining + numValidation);

        // Our training set will be the first num_train points from the original training set.
        trainingSet = trainSet.getSubSet(1 , numTraining + 1);

        // Development set - a small subset of the training set.
        devSet = trainingSet.getSubSet(1, numDev + 1); // TODO change to random; mask = np.random.choice(num_training, num_dev, replace=False)

        // The first numTest points of the original test set as the testing set.
        testingSet = testSet.getSubSet(1, numTest + 1);
    }

    private CifarDataSet loadFromFiles(Collection<String> files) {
        INDArray dataSet = Nd4j.create(IMAGES_IN_FILE * files.size(), IMAGE_LEN);
        INDArray dataSetlabels = Nd4j.create(IMAGES_IN_FILE * files.size(), 1);
        int imgCount = 0;
        try {
            byte[] buffer = new byte[IMAGE_LEN];
            for (String fileName : files) {
                FileInputStream fis = new FileInputStream(fileName);
                for (int j = 0 ; j < IMAGES_IN_FILE ; j++) {
                    dataSetlabels.putScalar(imgCount, fis.read());
                    if (fis.read(buffer) != -1) {
                        for (int i = 0 ; i < IMAGE_LEN ; i += 3) {
                            int pos = i / 3;
                            dataSet.putScalar(imgCount, i, buffer[pos] & 0xFF);
                            dataSet.putScalar(imgCount, i + 1, buffer[pos + 1024] & 0xFF);
                            dataSet.putScalar(imgCount, i + 2, buffer[pos + 2048] & 0xFF);
                        }
                        imgCount++;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new CifarDataSet(dataSet, dataSetlabels);
    }

    private void loadProperties() {
        Properties prop = new Properties();
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();

        try(InputStream input = classloader.getResourceAsStream("cifar.properties")) {

            prop.load(input);

            String path = prop.getProperty("path");
            String[] data = prop.getProperty("data").split(",");
            String[] test = prop.getProperty("test").split(",");

            for (String s : data) {
                dataFiles.add(path + "/" + s.trim());
            }
            for (String s : test) {
                testFiles.add(path + "/" + s.trim());
            }

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    // Getters & Setters ============================================
    public CifarDataSet getTrainingSet() {
        return trainingSet;
    }

    public CifarDataSet getValidationSet() {
        return validationSet;
    }

    public CifarDataSet getDevSet() {
        return devSet;
    }

    public CifarDataSet getTestingSet() {
        return testingSet;
    }

}
