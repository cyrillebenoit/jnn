import org.jblas.DoubleMatrix;

public class NeuralNetwork {
    private int inputs;
    private int hidden;
    private int outputs;
    private DoubleMatrix[] matrices;
    private DoubleMatrix[] activations;
    private double learningRate;

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

    private void initializeRandomMatrix() {
        this.matrices[0] = DoubleMatrix.rand(this.hidden, this.inputs + 1);
        for (int pointer = 1; pointer < this.matrices.length - 1; pointer++) {
            this.matrices[pointer] = DoubleMatrix.rand(this.hidden, this.hidden + 1);
        }
        this.matrices[this.matrices.length - 1] = DoubleMatrix.rand(this.outputs, this.hidden + 1);
    }

    public double[] getRandomInputs() {
        return DoubleMatrix.rand(inputs).toArray();
    }

    public double[] guess(double[] inputs) throws NeuralNetworkException {
        if (inputs.length != this.inputs)
            throw new NeuralNetworkException("Wrong inputs length.");
        DoubleMatrix res = new DoubleMatrix(inputs);
        return guess(res);
    }

    private double[] guess(DoubleMatrix inputs) throws NeuralNetworkException {
        if (inputs.columns != 1 || inputs.rows != this.inputs)
            throw new NeuralNetworkException("Wrong inputs dimension.");
        this.activations[0] = inputs;
        for (int pointer = 0; pointer < this.matrices.length; pointer++) {
            inputs = DoubleMatrix.concatVertically(inputs, DoubleMatrix.ones(1));
            inputs = this.matrices[pointer].mmul(inputs);
            inputs = MatrixFunctions.sigmoid(inputs);
            this.activations[pointer + 1] = inputs;
        }
        return inputs.toArray();
    }

    private DoubleMatrix _guess(DoubleMatrix inputs) throws NeuralNetworkException {
        if (inputs.columns != 1 || inputs.rows != this.inputs)
            throw new NeuralNetworkException("Wrong inputs dimension.");
        this.activations[0] = inputs;
        for (int pointer = 0; pointer < this.matrices.length; pointer++) {
            inputs = DoubleMatrix.concatVertically(inputs, DoubleMatrix.ones(1));
            inputs = this.matrices[pointer].mmul(inputs);
            inputs = MatrixFunctions.sigmoid(inputs);
            this.activations[pointer + 1] = inputs;
        }
        return inputs;
    }

    public void train(double[] inputs, double[] answer) throws NeuralNetworkException {
        if (inputs.length != this.inputs || answer.length != this.outputs)
            throw new NeuralNetworkException("Wrong lengths.");
        train(new DoubleMatrix(inputs), new DoubleMatrix(answer));
    }

    private void train(DoubleMatrix inputs, DoubleMatrix answer) throws NeuralNetworkException {
        if (inputs.rows != this.inputs || inputs.columns != 1 || answer.rows != this.outputs || answer.columns != 1)
            throw new NeuralNetworkException("Wrong dimensions.");
        DoubleMatrix guess = this._guess(inputs);
        DoubleMatrix outputErrors = answer.sub(guess);
        DoubleMatrix[] changes = new DoubleMatrix[this.matrices.length];
        // backward propagation
        for (int layer = this.activations.length - 1; layer > 0; layer--) {
            // for each layer (starting from the end)
            DoubleMatrix weights_ = this.matrices[layer - 1];
            inputs = this.activations[layer - 1];
            DoubleMatrix outputs_ = this.activations[layer];
            // calculate gradient for weights and biases nudges
            DoubleMatrix gradientProduct = MatrixFunctions.dsigmoid(new DoubleMatrix(outputs_.toArray()));
            DoubleMatrix gradient = gradientProduct.mul(outputErrors);
            DoubleMatrix delta = gradient.mmul(inputs.transpose());
            delta = DoubleMatrix.concatHorizontally(delta, gradient);
            // This is what should change for both the weights and the biases for this layer
            changes[layer - 1] = delta;

            // calculate error of previous layer
            DoubleMatrix input_errors = new DoubleMatrix(inputs.rows);
            for (int input_neuron = 0; input_neuron < inputs.length; input_neuron++) {
                double input_error = 0;
                for (int output = 0; output < outputs_.length; output++) {
                    // This part is commented out since it should be used to be more precise,
                    // but when enabled, learning process gets stuck 99 % of the time
                    // output_weights = 0
                    // for inp in range(0, len(inputs_), 1):
                    // output_weights += matrix_layer_[inp][output]

                    double partial_error = outputErrors.get(output) * weights_.get(output, input_neuron);
                    // partial_error /= output_weights
                    input_error += partial_error;
                }
                // Add this input error to the array of errors in the previous layer
                input_errors.put(input_neuron, input_error);
                // Use input error as output error for next loop
            }
            outputErrors = input_errors;
        }
        for (int i = 0; i < this.matrices.length; i++)
            this.matrices[i] = this.matrices[i].add(changes[i].mul(this.learningRate));
    }
}
