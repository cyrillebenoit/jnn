import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetwork {
    private int inputs;
    private int hidden;
    private int outputs;
    private DoubleMatrix[] matrices;
    private DoubleMatrix[] activations;
    private double learningRate;

    /**
     * Main constructor.
     *
     * @param inputs       number of input neurons
     * @param hidden       number of neurons per hidden layer
     * @param hiddenLayers number of hidden layers
     * @param outputs      number of output neurons
     * @throws NeuralNetworkException if one of the parameters is less than 1
     */
    public NeuralNetwork(int inputs, int hidden, int hiddenLayers, int outputs) throws NeuralNetworkException {
        if (inputs < 1 || hidden < 1 || hiddenLayers < 1 || outputs < 1)
            throw new NeuralNetworkException("Lol.");
        this.inputs = inputs;
        this.hidden = hidden;
        this.outputs = outputs;
        this.matrices = new DoubleMatrix[hiddenLayers + 1];
        this.activations = new DoubleMatrix[hiddenLayers + 2];
        this.learningRate = 1;
        this.initializeRandomMatrix();
    }

    /**
     * Learning rate setter.
     *
     * @param learningRate new learning rate value
     */
    public void setLearningRate(double learningRate) {
        if (learningRate > 0)
            this.learningRate = learningRate;
    }

    /**
     * This function initializes the weights and biases to random values.
     */
    private void initializeRandomMatrix() {
        this.matrices[0] = DoubleMatrix.rand(this.hidden, this.inputs + 1);
        for (int pointer = 1; pointer < this.matrices.length - 1; pointer++) {
            this.matrices[pointer] = DoubleMatrix.rand(this.hidden, this.hidden + 1);
        }
        this.matrices[this.matrices.length - 1] = DoubleMatrix.rand(this.outputs, this.hidden + 1);
    }

    /**
     * This function returns a random list of inputs.
     *
     * @return double[] random list of inputs.
     */
    public double[] getRandomInputs() {
        return DoubleMatrix.rand(inputs).toArray();
    }

    /**
     * This function is used to get a prediction from a list of inputs
     *
     * @param inputs list of input neurons activations
     * @return list of outputs activations
     * @throws NeuralNetworkException if inputs are not of the right length.
     */
    public double[] guess(double[] inputs) throws NeuralNetworkException {
        if (inputs.length != this.inputs)
            throw new NeuralNetworkException("Wrong inputs length.");
        DoubleMatrix res = new DoubleMatrix(inputs);
        return guess(res).toArray();
    }


    /**
     * This function is used to get a prediction from a matrix of inputs
     *
     * @param inputs matrix of input neurons activations
     * @return matrix of outputs activations
     * @throws NeuralNetworkException if inputs are not of the right dimension.
     */
    private DoubleMatrix guess(DoubleMatrix inputs) throws NeuralNetworkException {
        if (inputs.columns != 1 || inputs.rows != this.inputs)
            throw new NeuralNetworkException("Wrong inputs dimension.");
        this.activations[0] = inputs;
        for (int pointer = 0; pointer < this.matrices.length; pointer++) {
            inputs = DoubleMatrix.concatVertically(inputs, DoubleMatrix.ones(1));
            inputs = this.matrices[pointer].mmul(inputs);
            inputs = MyMatrixFunctions.sigmoid(inputs);
            this.activations[pointer + 1] = inputs;
        }
        return inputs;
    }

    /**
     * This function is used to train the neural network from a list of inputs and an expected output.
     *
     * @param inputs list of inputs
     * @param answer list of expected outputs
     * @throws NeuralNetworkException if inputs or answer is wrongly dimensioned.
     */
    public void train(double[] inputs, double[] answer) throws NeuralNetworkException {
        if (inputs.length != this.inputs || answer.length != this.outputs)
            throw new NeuralNetworkException("Wrong lengths.");
        train(new DoubleMatrix(inputs), new DoubleMatrix(answer));
    }

    /**
     * This function is used to train the neural network from a matrix of inputs and an expected output.
     *
     * @param inputs matrix of inputs
     * @param answer matrix of expected outputs
     * @throws NeuralNetworkException if inputs or answer is wrongly dimensioned.
     */
    private void train(DoubleMatrix inputs, DoubleMatrix answer) throws NeuralNetworkException {
        if (inputs.rows != this.inputs || inputs.columns != 1 || answer.rows != this.outputs || answer.columns != 1)
            throw new NeuralNetworkException("Wrong dimensions.");
        DoubleMatrix guess = this.guess(inputs);
        DoubleMatrix outputErrors = answer.sub(guess);
        DoubleMatrix[] changes = new DoubleMatrix[this.matrices.length];
        // backward propagation
        for (int layer = this.activations.length - 1; layer > 0; layer--) {
            // for each layer (starting from the end)
            DoubleMatrix weights_ = this.matrices[layer - 1];
            inputs = this.activations[layer - 1];
            DoubleMatrix outputs_ = this.activations[layer];
            // calculate gradient for weights and biases nudges
            DoubleMatrix gradientProduct = MyMatrixFunctions.dsigmoid(new DoubleMatrix(outputs_.toArray()));
            DoubleMatrix gradient = gradientProduct.mul(outputErrors);
            DoubleMatrix delta = gradient.mmul(inputs.transpose());
            delta = DoubleMatrix.concatHorizontally(delta, gradient);
            // This is what should change for both the weights and the biases for this layer
            changes[layer - 1] = delta;

            // calculate error of previous layer
            DoubleMatrix input_errors = new DoubleMatrix(inputs.rows);
            DoubleMatrix rowsSums = MatrixFunctions.abs(weights_).rowSums().sub(MatrixFunctions.abs(weights_).getColumn(weights_.columns - 1));
            for (int input_neuron = 0; input_neuron < inputs.length; input_neuron++) {
                double input_error = 0;
                for (int output = 0; output < outputs_.length; output++) {
                    // Distribute the error backwards proportionally to the weights.
                    double proportion = Math.abs(weights_.get(output, input_neuron)) / rowsSums.get(output);
                    input_error += outputErrors.get(output) * proportion;
                }
                // Add this input error to the array of errors in the previous layer
                input_errors.put(input_neuron, input_error);
            }
            // Use input error as output error for next loop
            outputErrors = input_errors;
        }
        for (int i = 0; i < this.matrices.length; i++)
            this.matrices[i] = this.matrices[i].add(changes[i].mul(this.learningRate));
    }
}
