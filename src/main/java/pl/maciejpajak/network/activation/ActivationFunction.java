package pl.maciejpajak.network.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Activation function interface.
 * Actually very similar to org.nd4j.linalg.activations.IActivation (but simpler).
 */
public interface ActivationFunction {

    /**
     * Calculates activation on input array.
     * @param input - input array.
     * @return - activation array.
     */
    INDArray process(INDArray input);

    /**
     * Backpropagation.
     * @param input     input before applying the function.
     * @param gradient  gradient to be backpropagated.
     * @return          backpropagated gradient.
     */
    INDArray backprop(INDArray input, INDArray gradient);

}
